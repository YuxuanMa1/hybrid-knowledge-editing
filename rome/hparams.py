"""
ROME_new: hyperparameters and layer selection
"""

import torch


class ROMEHyperParams:
    def __init__(self, layer, hidden_size, cproj_module, mlp_module):
        self.layer = layer
        self.hidden_size = hidden_size
        self.cproj_module = cproj_module
        self.mlp_module = mlp_module


def auto_select_layer(model_name: str):
    """
    Auto layer mapping based on GPT-2 size.
    """
    if model_name == "gpt2":
        return 6
    elif model_name == "gpt2-medium":
        return 10
    elif model_name == "gpt2-large":
        return 15
    elif model_name == "gpt2-xl":
        return 20
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def get_hparams(model_name: str, model):
    """
    Determine hidden size + layer name + module names.
    """

    hidden = model.config.hidden_size
    layer = auto_select_layer(model_name)

    # GPT-2 structure:
    # transformer.h[layer].mlp.c_proj
    cproj = f"transformer.h.{layer}.mlp.c_proj"
    mlp = f"transformer.h.{layer}.mlp"

    return ROMEHyperParams(
        layer=layer,
        hidden_size=hidden,
        cproj_module=cproj,
        mlp_module=mlp
    )
