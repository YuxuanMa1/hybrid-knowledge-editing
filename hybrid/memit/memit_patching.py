import torch
from memit.memit_utils import print_shape


def apply_delta_to_layer(model, layer_id, delta, alpha=0.1, verbose=False):
    mlp = model.transformer.h[layer_id].mlp
    c_proj = mlp.c_proj

    delta_T = delta.T

    with torch.no_grad():
        c_proj.weight += alpha * delta_T

    if verbose:
        print(f"[Layer {layer_id}] ΔW applied.")


def apply_memit_patch_multi(model, deltas, alpha=0.1, verbose=False):
    for layer_id, delta in deltas.items():
        apply_delta_to_layer(
            model,
            layer_id,
            delta,
            alpha=alpha,
            verbose=verbose
        )
    return model
