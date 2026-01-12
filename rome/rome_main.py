"""
ROME_new main edit logic
"""

import torch
from .hparams import get_hparams
from .compute_u import compute_u
from .compute_v import compute_v
from .util.nethook import get_parameter, set_parameter


def apply_rome_to_model(model, tokenizer, request):
    """
    request = {
        "subject": "...",
        "target":  "...",
        "prompt":  "..."
    }
    """

    subject = request["subject"]
    target = request["target"]

    # determine layer and modules
    hparams = get_hparams("gpt2", model)

    print(f"\n[ROME_new] Editing layer {hparams.layer} subject={subject} target={target}")

    # 1. compute u
    u = compute_u(model, tokenizer, subject, hparams)

    # 2. compute v
    v = compute_v(model, tokenizer, subject, target, hparams)

    # 3. compute ΔW
    cproj = hparams.cproj_module
    W = get_parameter(model, cproj)

    deltaW = torch.ger(v, u)     # (out_dim, in_dim)
    new_W = W + deltaW

    set_parameter(model, cproj, new_W)

    print("[ROME_new] Edit applied successfully.")
    return model, {"layer": hparams.layer}

