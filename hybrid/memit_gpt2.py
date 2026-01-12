import torch


###############################################
# Utility
###############################################

def normalize(x, eps=1e-8):
    return x / (x.norm() + eps)


###############################################
# GPT-2 Model Info
###############################################

def get_gpt2_info(model):
    block = model.transformer.h[0].mlp
    hidden = block.c_proj.weight.shape[1]        # 1280
    intermediate = block.c_proj.weight.shape[0]  # 5120
    num_layers = len(model.transformer.h)
    return hidden, intermediate, num_layers


###############################################
# K* (from c_fc)
###############################################

def compute_kstar(model, tok, prompts, layer):
    """multiple prompts → average k*"""
    ks = []
    for p in prompts:
        inp = tok(p, return_tensors="pt")
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True, return_dict=True)
        h_prev = out.hidden_states[layer][0, -1]
        mlp = model.transformer.h[layer].mlp
        k = mlp.c_fc(h_prev)  # intermediate dim
        ks.append(k)

    k_avg = torch.stack(ks, dim=0).mean(dim=0)
    return normalize(k_avg)


###############################################
# V* (from target token embedding)
###############################################

def compute_vstar(model, tok, targets):
    ids = tok(targets, add_special_tokens=False)["input_ids"]
    flat = [i for seq in ids for i in seq]

    embed = model.transformer.wte
    v = embed(torch.tensor(flat)).mean(dim=0)
    return normalize(v)


###############################################
# MEMIT Multi-layer Update
###############################################

def memit_edit(model, tok, prompts, targets, layers, lr=0.2):

    v_star = compute_vstar(model, tok, targets)

    for layer in layers:
        k_star = compute_kstar(model, tok, prompts, layer)

        mlp = model.transformer.h[layer].mlp
        c_proj = mlp.c_proj

        # c_proj.weight: (intermediate, hidden)
        W = c_proj.weight.T  # → (hidden, intermediate)

        old_v = W @ k_star  # hidden

        resid = (v_star - old_v).unsqueeze(1)
        k_row = k_star.unsqueeze(0)

        delta = resid @ k_row  # (hidden, intermediate)

        # Update
        c_proj.weight.data += lr * delta.T

    return model


###############################################
# PUBLIC API
###############################################

def apply_memit(model, tok, prompt, target, lr=0.2):
    hidden, inter, n_layers = get_gpt2_info(model)

    # auto-layer selection
    if n_layers == 12:    # GPT-2 small
        layers = [4, 6, 8]
    elif n_layers == 24:  # Medium
        layers = [8, 12, 16, 20]
    elif n_layers == 36:  # Large
        layers = [12, 18, 24, 30]
    else:                 # 48 (XL)
        layers = [15, 20, 25, 30, 35, 40]

    return memit_edit(
        model,
        tok,
        prompts=[prompt],
        targets=[target],
        layers=layers,
        lr=lr,
    )

