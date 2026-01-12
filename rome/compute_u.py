"""
Compute left vector u for ROME_new
"""

import torch
from .repr_tools import get_mlp_input_vector, make_context_sentences, normalize


def compute_u(model, tokenizer, subject, hparams):
    ctxs = make_context_sentences(subject)

    vecs = []
    for s in ctxs:
        v = get_mlp_input_vector(
            model=model,
            tokenizer=tokenizer,
            text=s,
            layer=hparams.layer,
            subject=subject
        )
        vecs.append(v)

    u = torch.stack(vecs, dim=0).mean(dim=0)
    return normalize(u)
