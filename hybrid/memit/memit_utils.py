import torch
import torch.nn.functional as F


############################################################
# Normalization utilities
############################################################

def normalize(vec, eps=1e-8):
    """L2 normalize vector safely."""
    return vec / (torch.norm(vec) + eps)


############################################################
# Token utilities
############################################################

def last_token_hidden(hidden_states, attn_mask=None):
    """
    Returns hidden state of last non-padding token.
    hidden_states: [batch, seq, hidden]
    attn_mask: [batch, seq]
    """
    if attn_mask is None:
        return hidden_states[:, -1, :]

    idxs = attn_mask.sum(dim=1) - 1
    out = []
    for i, idx in enumerate(idxs):
        out.append(hidden_states[i, idx, :])
    return torch.stack(out, dim=0)


############################################################
# Shape helpers
############################################################

def flatten_first_two(t):
    """(a, b, ...) → (a*b, ...)"""
    return t.reshape(-1, *t.shape[2:])


def unflatten_first(t, a, b):
    """(a*b, ...) → (a, b, ...)"""
    return t.reshape(a, b, *t.shape[1:])


############################################################
# MLP weight helpers
############################################################

def get_cproj_W(model, layer_id):
    """
    Returns:
       W^T = c_proj.weight.T   (shape: hidden x intermediate)
    """
    block = model.transformer.h[layer_id].mlp
    W = block.c_proj.weight.T
    return W


def get_fc_forward(model, layer_id, h):
    """
    Returns c_fc(h)  (intermediate dim)
    """
    block = model.transformer.h[layer_id].mlp
    return block.c_fc(h)


############################################################
# Jacobian utilities
############################################################

def compute_pseudoinverse(A, lamb=0.01):
    """
    Computes Moore-Penrose pseudoinverse with Tikhonov regularization:
         A^+ = (A^T A + λI)^(-1) A^T
    A: [m, n]
    Return: [n, m]
    """
    # A^T A
    ATA = A.T @ A

    # add regularization
    reg = lamb * torch.eye(ATA.shape[0], device=A.device)

    inv = torch.inverse(ATA + reg)
    return inv @ A.T


############################################################
# Debugging helpers
############################################################

def print_shape(name, tensor):
    print(f"{name}: shape={tuple(tensor.shape)}")
