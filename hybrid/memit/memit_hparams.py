from dataclasses import dataclass
from typing import List, Optional


@dataclass
class MEMITHyperParams:
    """
    Hyperparameters controlling the MEMIT editing behavior.
    This is the GPT-2 specialized version:
      - Supports GPT-2 small / medium / large / XL
      - No GPT-J / GPT-NeoX branches kept
    """

    # Which layers to edit
    layers: List[int]

    # scale for delta W (alpha in paper)
    edit_lr: float = 0.2

    # How many prompts to use for K* averaging
    n_calibration_prompts: int = 1

    # Whether to normalize k* / v*
    normalize_k: bool = True
    normalize_v: bool = True

    # If true, MEMIT will use pseudoinverse-based Jacobian solving
    use_jacobian: bool = True

    # for stability in pseudoinverse
    pinv_lambda: float = 0.01

    # max token length used in Jacobian extraction
    max_jacobian_tokens: int = 40

    # DEBUG: print shapes / loss etc.
    verbose: bool = False

    # ==================================================
    # GPT-2 model internal dimensions (auto-filled later)
    # ==================================================
    d_model: Optional[int] = None            # hidden size
    d_intermediate: Optional[int] = None     # intermediate size
    n_layers: Optional[int] = None


def get_default_hparams(model):
    """
    Creates hyperparams based on GPT-2 model architecture.
    """
    n_layers = len(model.transformer.h)
    d_model = model.transformer.h[0].mlp.c_proj.weight.shape[1]
    d_inter = model.transformer.h[0].mlp.c_proj.weight.shape[0]

    # Auto-select recommended layers for GPT-2 sizes
    if n_layers == 12:   # GPT-2 small
        layers = [5, 7, 9]
    elif n_layers == 24:  # GPT-2 medium
        layers = [6, 12, 18]
    elif n_layers == 36:  # GPT-2 large
        layers = [10, 18, 26, 32]
    else:                 # GPT-2 XL (48 layers)
        layers = [12, 18, 24, 30, 36, 42]

    return MEMITHyperParams(
        layers=layers,
        edit_lr=0.15,
        n_calibration_prompts=1,
        normalize_k=True,
        normalize_v=True,
        use_jacobian=True,
        pinv_lambda=0.01,
        d_model=d_model,
        d_intermediate=d_inter,
        n_layers=n_layers,
    )
