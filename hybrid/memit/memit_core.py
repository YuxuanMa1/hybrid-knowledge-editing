import torch

from memit.memit_utils import normalize, get_fc_forward
from memit.memit_jacobian import (
    compute_jacobian,
    solve_dk_from_dv,
    compute_delta_tensor,
)
from memit.memit_patching import apply_memit_patch_multi


def compute_kstar_multi(model, tok, prompts, layer_id, normalize_k=True):
    ks = []
    for p in prompts:
        inp = tok(p, return_tensors="pt")
        out = model(
            **inp,
            output_hidden_states=True,
            return_dict=True
        )
        h_prev = out.hidden_states[layer_id][0, -1]
        mlp = model.transformer.h[layer_id].mlp
        k = mlp.c_fc(h_prev)
        ks.append(k.detach())

    k_star = torch.stack(ks).mean(dim=0)
    return normalize(k_star) if normalize_k else k_star


def compute_vstar_multi(model, tok, targets, normalize_v=True):
    ids = tok(targets, add_special_tokens=False)["input_ids"]
    flat = [i for seq in ids for i in seq]
    embed = model.transformer.wte
    v = embed(torch.tensor(flat)).mean(dim=0)
    return normalize(v) if normalize_v else v


def assemble_layer_deltas(
    model,
    tok,
    prompts,
    targets,
    layers,
    hparams
):
    deltas = {}
    verbose = hparams.verbose

    v_star = compute_vstar_multi(
        model,
        tok,
        targets,
        normalize_v=hparams.normalize_v
    ).detach()

    for L in layers:
        if verbose:
            print(f"\n=== Computing for Layer {L} ===")

        k_star = compute_kstar_multi(
            model,
            tok,
            prompts,
            L,
            normalize_k=hparams.normalize_k
        ).detach()

        if hparams.use_jacobian:
            J = compute_jacobian(
                model,
                tok,
                prompts[0],
                layer_id=L,
                max_tokens=hparams.max_jacobian_tokens
            )
        else:
            raise NotImplementedError("Jacobian-free MEMIT not supported.")

        mlp = model.transformer.h[L].mlp
        W = mlp.c_proj.weight.T
        old_v = W @ k_star
        dv = (v_star - old_v).detach()

        dk = solve_dk_from_dv(
            J,
            dv,
            lamb=hparams.pinv_lambda,
            verbose=verbose
        )

        k_new = k_star + dk
        delta = compute_delta_tensor(model, L, k_new, v_star)

        deltas[L] = delta.detach()

    return deltas


def run_memit_edit(
    model,
    tok,
    prompts,
    targets,
    hparams
):

    deltas = assemble_layer_deltas(
        model,
        tok,
        prompts,
        targets,
        layers=hparams.layers,
        hparams=hparams
    )

    model = apply_memit_patch_multi(
        model,
        deltas,
        alpha=hparams.edit_lr,
        verbose=hparams.verbose
    )

    return model

