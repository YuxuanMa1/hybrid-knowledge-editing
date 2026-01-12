import torch

from memit.memit_hparams import (
    MEMITHyperParams,
    get_default_hparams,
)
from memit.memit_core import run_memit_edit


def apply_memit(
    model,
    tokenizer,
    prompt,
    target,
    hparams=None,
):

    if hparams is None:
        hparams = get_default_hparams(model)

    prompts = [prompt]
    targets = [target]

    edited_model = run_memit_edit(
        model,
        tokenizer,
        prompts,
        targets,
        hparams
    )

    return edited_model

