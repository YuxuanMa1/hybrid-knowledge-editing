import json, math, torch, re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from pmet.pmet_editor import PMETEditor


# ============================================================
# subject extractor
# ============================================================
def extract_subject_from_prompt(prompt):
    p = prompt.replace("?", "").strip()
    m = re.search(r"of ([A-Za-z0-9\s\-]+)$", p)
    if m:
        return m.group(1).strip()
    return p.split()[-1]


# ============================================================
# Load model
# ============================================================
MODEL_NAME = "gpt2-large"  # 可改成 "gpt2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model.eval()

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id


# ============================================================
# Auto-select edit layers
# ============================================================
num_layers = model.config.n_layer
edit_layers = list(range(num_layers - 4, num_layers))
print("Auto-selected PMET layers:", edit_layers)

intermediate_size = (
    model.config.n_inner if model.config.n_inner else model.config.n_embd * 4
)


# ============================================================
# Initialize PMET
# ============================================================
pmet = PMETEditor(
    model,
    rank=32,
    edit_layers=edit_layers,
    hidden_size=model.config.n_embd,
    intermediate_size=intermediate_size,
    alpha=0.01               # crucial for GPT-2-large stability
)


# ============================================================
# Load dataset
# ============================================================
data_path = Path("../data/counterfact.jsonl")
dataset = [json.loads(line) for line in data_path.open("r", encoding="utf-8")]
print("Loaded", len(dataset), "samples.")


# ============================================================
# Predict
# ============================================================
def predict_answer(question):
    prompt = f"Q: {question}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return full.split("A:")[-1].strip()


def calc_ppl(text):
    if not text.strip():
        return 999.0
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    if torch.isnan(loss) or torch.isinf(loss):
        return 999.0
    return math.exp(loss.item())


# ============================================================
# Run PMET
# ============================================================
succ_list, port_list, loc_list, flu_list = [], [], [], []

for i, item in enumerate(dataset, 1):

    q = item["prompt"]
    gold = item["answers"]
    subject = extract_subject_from_prompt(q)
    new_obj = gold[0]

    print(f"\n===== PMET Editing {i} =====")
    print(f"Subject = {subject} | New Object = {new_obj}")

    pmet.apply_edit(tokenizer, subject, new_obj)

    # ---- Success ----
    pred = predict_answer(q)
    ed_succ = any(g.lower() in pred.lower() for g in gold)
    succ_list.append(ed_succ)
    print("Pred:", pred)
    print("Edit Success:", ed_succ)

    # ---- Portability ----
    ports = []
    for pq in item.get("portability_prompts", []):
        p_pred = predict_answer(pq)
        ok = any(g.lower() in p_pred.lower() for g in gold)
        ports.append(ok)

    if ports:
        avg_port = sum(ports)/len(ports)
        port_list.append(avg_port)
        print("Portability:", avg_port)

    # ---- Locality ----
    locs = []
    for LQ, LA in zip(item.get("locality_prompts", []), item.get("locality_answers", [])):
        l_pred = predict_answer(LQ)
        locs.append(LA.lower() in l_pred.lower())

    if locs:
        avg_loc = sum(locs)/len(locs)
        loc_list.append(avg_loc)
        print("Locality:", avg_loc)

    # ---- Fluency ----
    flu = calc_ppl(pred)
    flu_list.append(flu)
    print("Fluency PPL:", flu)


# ============================================================
# Summary
# ============================================================
print("\n=========== PMET SUMMARY ===========")
print(f"Edit Success: {sum(succ_list)/len(succ_list)*100:.1f}%")
if port_list:
    print(f"Portability:  {sum(port_list)/len(port_list)*100:.1f}%")
if loc_list:
    print(f"Locality:     {sum(loc_list)/len(loc_list)*100:.1f}%")
print(f"Avg PPL:       {sum(flu_list)/len(flu_list):.2f}")
print("====================================")
