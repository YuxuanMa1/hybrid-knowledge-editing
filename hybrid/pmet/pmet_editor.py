import torch
import torch.nn as nn


class PMETEditor(nn.Module):
    """
    PMET Editor (Stable Version)
    - Supports GPT-2, GPT-2-medium, GPT-2-large
    - Correct Δ_fc / Δ_proj sizes:
        c_fc:   [h, m]
        c_proj: [m, h]
    - Includes stability fixes (clamp, norm, scale)
    """

    def __init__(
        self,
        model,
        rank=32,
        edit_layers=None,
        hidden_size=768,
        intermediate_size=3072,
        alpha=0.01         # recommended scaling for large models
    ):
        super().__init__()

        self.model = model
        self.rank = rank
        self.edit_layers = edit_layers
        self.h = hidden_size
        self.m = intermediate_size
        self.alpha = alpha

        # -----------------------------
        # Low-rank ΔW decomposition
        # -----------------------------
        # c_fc: [h, m]
        self.U_fc = nn.Parameter(torch.randn(len(edit_layers), self.h, rank) * 0.02)
        self.V_fc = nn.Parameter(torch.randn(len(edit_layers), self.m, rank) * 0.02)

        # c_proj: [m, h]
        self.U_proj = nn.Parameter(torch.randn(len(edit_layers), self.m, rank) * 0.02)
        self.V_proj = nn.Parameter(torch.randn(len(edit_layers), self.h, rank) * 0.02)

        # gate(g)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, rank)
        )

        # per-layer scaling
        self.layer_gate = nn.Parameter(torch.ones(len(edit_layers)))


    # --------------------------------------------------------
    # Word encoder
    # --------------------------------------------------------
    def encode_word(self, tokenizer, text):
        ids = tokenizer.encode(text, return_tensors="pt")
        with torch.no_grad():
            emb = self.model.transformer.wte(ids)[0]   # [seq, h]
        return emb.mean(dim=0)  # [h]


    # --------------------------------------------------------
    # Apply PMET edit
    # --------------------------------------------------------
    def apply_edit(self, tokenizer, subject, new_object, context=None):

        subj_vec = self.encode_word(tokenizer, subject)
        obj_vec  = self.encode_word(tokenizer, new_object)
        ctx_vec = subj_vec if context is None else self.encode_word(tokenizer, context)

        descriptor = torch.cat([subj_vec, obj_vec, ctx_vec], dim=-1)

        # ---- Stable g ----
        g = self.gate(descriptor)
        g = g / (g.norm() + 1e-6)     # normalize gate
        g_diag = torch.diag(g)

        # ============================================================
        # Update each target Transformer layer
        # ============================================================
        for li, layer_id in enumerate(self.edit_layers):

            # ------------------- Build Δ_fc: [h, m] -------------------
            Ufc = self.U_fc[li]      # [h, r]
            Vfc = self.V_fc[li]      # [m, r]
            delta_fc = Ufc @ g_diag @ Vfc.t()   # [h, m]

            # stability clamp + scale
            delta_fc = torch.clamp(delta_fc, -1.0, 1.0) * self.alpha
            delta_fc[torch.isnan(delta_fc)] = 0
            delta_fc[torch.isinf(delta_fc)] = 0

            W_fc = self.model.transformer.h[layer_id].mlp.c_fc.weight.data
            W_fc.add_(delta_fc.to(W_fc.device))


            # ------------------- Build Δ_proj: [m, h] -------------------
            Upr = self.U_proj[li]    # [m, r]
            Vpr = self.V_proj[li]    # [h, r]
            delta_proj = Upr @ g_diag @ Vpr.t()  # [m, h]

            delta_proj = torch.clamp(delta_proj, -1.0, 1.0) * self.alpha
            delta_proj[torch.isnan(delta_proj)] = 0
            delta_proj[torch.isinf(delta_proj)] = 0

            W_proj = self.model.transformer.h[layer_id].mlp.c_proj.weight.data
            W_proj.add_(delta_proj.to(W_proj.device))

        return g.detach()



