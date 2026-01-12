import torch
from memit.memit_utils import (
    normalize,
    get_cproj_W,
    get_fc_forward,
    last_token_hidden,
    compute_pseudoinverse,
    print_shape,
)


def get_layer_mlp(model, layer_id):
    return model.transformer.h[layer_id].mlp


def compute_jacobian(model, tok, prompt, layer_id, max_tokens=40):
    inp = tok(prompt, return_tensors="pt")
    out = model(
        **inp,
        output_hidden_states=True,
        return_dict=True
    )

    h_prev = out.hidden_states[layer_id][0, -1]
    h_prev = h_prev.detach().clone().requires_grad_(True)

    mlp = get_layer_mlp(model, layer_id)
    fc_out = mlp.c_fc(h_prev)
    proj_out = mlp.c_proj(fc_out)

    hidden_dim = proj_out.shape[0]
    inter_dim = fc_out.shape[0]

    J = []
    for i in range(hidden_dim):
        grad = torch.autograd.grad(
            proj_out[i],
            fc_out,
            retain_graph=True,
        )[0]
        J.append(grad.detach())

    J = torch.stack(J, dim=0)
    return J


def solve_dk_from_dv(J, dv, lamb=0.01, verbose=False):
    J_pinv = compute_pseudoinverse(J, lamb=lamb)
    dk = J_pinv @ dv
    return dk


def compute_delta_tensor(model, layer_id, k_star, v_star):
    W = get_cproj_W(model, layer_id)
    old_v = W @ k_star
    resid = (v_star - old_v).unsqueeze(1)
    k_row = k_star.unsqueeze(0)
    delta = resid @ k_row
    return delta
