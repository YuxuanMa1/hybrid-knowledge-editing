"""
ROME_new: representation tools
"""

import torch
from .util.nethook import TraceDict


def locate_subject(tokenizer, text, subject):
    """
    Locate subject token position robustly under GPT-2 BPE tokenization.
    """
    enc_text = tokenizer(text)["input_ids"]
    enc_subj = tokenizer(subject)["input_ids"]

    Ls = len(enc_subj)

    # simple sliding window search
    for i in range(len(enc_text) - Ls + 1):
        if enc_text[i:i+Ls] == enc_subj:
            return i, i + Ls - 1

    return None, None


def get_mlp_input_vector(model, tokenizer, text, layer, subject):
    """
    Hooks into transformer.h[layer].mlp to collect input activations.
    """

    module_name = f"transformer.h.{layer}.mlp"

    with TraceDict(
        model,
        [module_name],
        retain_input=True,
        retain_output=False
    ) as traces:

        inputs = tokenizer(text, return_tensors="pt")
        model(**inputs)

        mlp_input = traces[module_name].input[0]       # shape: (1, seq, hidden)
        mlp_input = mlp_input.squeeze(0)               # (seq, hidden)

        start, end = locate_subject(tokenizer, text, subject)
        if start is None:
            return mlp_input.mean(dim=0)

        return mlp_input[start:end+1].mean(dim=0)


def make_context_sentences(subject):
    """
    Stable context generator (unlike original ROME templates).
    """
    templates = [
        f"{subject} is a well known entity.",
        f"Many people discuss {subject} in various texts.",
        f"This sentence contains the subject {subject}.",
    ]
    return templates


def normalize(vec):
    return vec / (vec.norm() + 1e-8)
