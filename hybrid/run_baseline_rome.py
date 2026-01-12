import os
import sys
import json
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ======================================================
# 1. Add rome_new package path
# ======================================================
ROME_PARENT = r"D:\llm-edit"
sys.path.append(ROME_PARENT)
print("🔧 sys.path added:", ROME_PARENT)

from rome.rome_main import apply_rome_to_model


# ======================================================
# 2. Select model
# ======================================================
MODEL_NAME = "gpt2-large"   # 可改为 gpt2-large

print(f"🔹 Loading GPT model: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# 重要：不要 deepcopy
def load_fresh_model():
    m = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    m.config.pad_token_id = m.config.eos_token_id
    m.eval()
    return m


# ======================================================
# 3. Answer generator
# ======================================================
def answer(model, question):
    prompt = f"Q: {question}\nA:"
    enc = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    txt = tokenizer.decode(out[0], skip_special_tokens=True)
    if "A:" in txt:
        return txt.split("A:")[-1].strip()
    return txt.strip()


# ======================================================
# 4. Metrics
# ======================================================
def normalize(txt):
    return txt.lower().strip().replace(".", "").replace(",", "")

def contains(pred, gold_list):
    p = normalize(pred)
    return any(normalize(g) in p for g in gold_list)

def safe_ppl(question, answer, model):
    text = f"Q: {question}\nA: {answer}"
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc["input_ids"]
    labels = input_ids.clone()

    q_len = len(tokenizer(f"Q: {question}\nA:", return_tensors="pt")["input_ids"][0])
    labels[0, :q_len] = -100

    with torch.no_grad():
        loss = model(input_ids, labels=labels).loss
        return min(math.exp(loss.item()), 500.0)


# ======================================================
# 5. Load CounterFact dataset
# ======================================================
DATA_PATH = r"D:\llm-edit\data\counterfact.jsonl"
dataset = [json.loads(l) for l in open(DATA_PATH, "r", encoding="utf-8")]

print(f"\n📘 Loaded {len(dataset)} CounterFact items.\n")


# ======================================================
# 6. Extract subject
# ======================================================
import re

def extract_subject(prompt):
    p = prompt.strip().rstrip("?.! ")

    m = re.search(r"of ([A-Za-z0-9\- ]+)\??$", p)
    if m: return m.group(1)

    m = re.search(r"Who is the CEO of (.+)", p)
    if m: return m.group(1)

    m = re.search(r"When was ([A-Za-z0-9\- ]+) founded", p, re.I)
    if m: return m.group(1)

    toks = p.split()
    for t in reversed(toks):
        if t[0].isupper():
            return t.replace("?", "")

    return toks[-1]


# ======================================================
# 7. Main experiment loop
# ======================================================
successes = []
ports = []
localscores = []
fluencies = []

for i, sample in enumerate(dataset, 1):

    prompt = sample["prompt"]
    answers = sample["answers"]
    subject = extract_subject(prompt)
    target = answers[0]

    print("\n" + "="*70)
    print(f"🧩 Edit {i}: {prompt}")
    print(f"🎯 Target: {target}")
    print(f"📌 Subject: {subject}")

    # load fresh model (NO DEEPCOPY)
    model = load_fresh_model()

    # build request
    request = {
        "subject": subject,
        "target": target,
        "prompt": prompt
    }

    # apply ROME edit
    model, _ = apply_rome_to_model(model, tokenizer, request)

    # evaluate edit success
    pred = answer(model, prompt)
    succ = contains(pred, answers)
    successes.append(succ)

    print(f"💬 Predicted: {pred}")
    print(f"👉 Edit Success: {'YES' if succ else 'NO'}")

    # portability
    port_res = []
    for pq in sample.get("portability_prompts", []):
        pp = answer(model, pq)
        port_res.append(contains(pp, answers))
    if port_res:
        ports.append(sum(port_res) / len(port_res))
        print(f"🔁 Portability: {ports[-1]*100:.1f}%")

    # locality
    loc_res = []
    for lq, la in zip(sample.get("locality_prompts", []),
                      sample.get("locality_answers", [])):
        lp = answer(model, lq)
        loc_res.append(contains(lp, [la]))
    if loc_res:
        localscores.append(sum(loc_res) / len(loc_res))
        print(f"🎯 Locality: {localscores[-1]*100:.1f}%")

    # fluency
    ppl = safe_ppl(prompt, pred, model)
    fluencies.append(ppl)
    print(f"💬 Fluency (PPL): {ppl:.1f}")


# ======================================================
# 8. Summary
# ======================================================
print("\n==================== ROME SUMMARY ====================")
print(f"✨ Edit Success Rate: {sum(successes)/len(successes)*100:.1f}%")
if ports: print(f"🔁 Portability: {sum(ports)/len(ports)*100:.1f}%")
if localscores: print(f"🎯 Locality: {sum(localscores)/len(localscores)*100:.1f}%")
print(f"💬 Average Fluency (PPL): {sum(fluencies)/len(fluencies):.1f}")
print("======================================================")
