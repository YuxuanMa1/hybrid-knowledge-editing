
c_fc: torch.Size([768, 3072])
c_proj: torch.Size([3072, 768])





from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("gpt2")
print("c_fc:", m.transformer.h[5].mlp.c_fc.weight.shape)
print("c_proj:", m.transformer.h[5].mlp.c_proj.weight.shape)
