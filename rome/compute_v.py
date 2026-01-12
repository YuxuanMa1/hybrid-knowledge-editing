"""
ROME_new: compute right vector v for GPT-2 MLP.c_proj
Produces v of dimension 4*hidden so that ΔW matches c_proj (3072×768)
"""

import torch
import torch.nn as nn
from .repr_tools import normalize
from .util.nethook import TraceDict


def compute_v(model, tokenizer, subject, target, hparams):

    layer = hparams.layer
    cproj_name = hparams.cproj_module  # e.g., transformer.h.6.mlp.c_proj

    # REQUIRED DIMENSION
    OUT_DIM = 4 * hparams.hidden_size   # GPT2-small: 4*768 = 3072

    # v must match c_proj output dimension!
    v = torch.zeros(OUT_DIM, requires_grad=True)

    optimizer = torch.optim.Adam([v], lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    # tokenize target
    tgt_id = tokenizer(target, return_tensors="pt")["input_ids"][0, 0]

    for step in range(80):
        optimizer.zero_grad()

        # inject v into c_proj output during forward pass
        with TraceDict(model, [cproj_name], retain_input=False, retain_output=True) as trace:

            # run model with simple prompt
            inputs = tokenizer(f"{subject} is", return_tensors="pt")
            logits = model(**inputs).logits[:, -1]

            # supervision
            loss = loss_fn(logits, tgt_id.unsqueeze(0))

            loss.backward()

        # gradient update
        optimizer.step()

        # normalize to avoid explosion
        v.data = normalize(v.data)

    return v.detach()


