import json, math, torch, copy
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== IMPORT THE OFFICIAL MEMIT API =====
from memit.memit_editor import apply_memit


# ============================================================
# 1. Load GPT-2 model
# ============================================================
MODEL_NAME = "gpt2-large"
print(f"🔹 Loading GPT-2 model for MEMIT: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

tokenizer.pad_token = tokenizer.eos_token
model_base.config.pad_token_id = tokenizer.eos_token_id
model_base.eval()


# ============================================================
# 2. Load counterfact dataset
# ============================================================
data_path = Path("../data/counterfact.jsonl")

with open(data_path, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f if line.strip()]

print(f"🔹 Loaded {len(dataset)} CounterFact samples.\n")


# ============================================================
# 3. Helper functions (same as baseline ROME)
# ============================================================

def normalize(text):
    return text.strip().lower().replace(".", "").replace(",", "")


def contains(pred, gold_list):
    """Check predicted text contains any gold answer."""
    p = normalize(pred)
    return any(normalize(g) in p for g in gold_list)


def safe_extract_answer(text):
    """Extract the part after 'A:'; safe fallback."""
    if "A:" in text:
        t = text.split("A:", 1)[-1].strip()
        return t if t else "[EMPTY]"
    return text


def answer(model, q):
    """Generate answer for one QA."""
    prompt = f"Q: {q}\nA:"
    inp = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    full = tokenizer.decode(out[0], skip_special_tokens=True)
    return safe_extract_answer(full)


def safe_ppl(model, text):
    """Compute perplexity of output text."""
    if not text.strip():
        return 999.0

    inp = tokenizer(text, return_tensors="pt")
    if inp["input_ids"].size(1) == 0:
        return 999.0

    with torch.no_grad():
        loss = model(**inp, labels=inp["input_ids"]).loss

    return math.exp(loss.item())


# ============================================================
# 4. MAIN LOOP — evaluate all 100 CounterFact edits
# ============================================================

edit_success = []
port_scores = []
loc_scores = []
fluencies = []

for i, sample in enumerate(dataset, 1):
    q = sample["prompt"]
    ans = sample["answers"]

    print("=" * 75)
    print(f"🧩 Edit {i}: {q}")
    print(f"🎯 Target: {ans[0]}")

    # Make a fresh copy of the model
    model = copy.deepcopy(model_base)

    # === APPLY MEMIT ===
    model = apply_memit(
        model=model,
        tokenizer=tokenizer,
        prompt=q,
        target=ans[0],
    )

    # === MAIN prediction ===
    pred = answer(model, q)
    ok = contains(pred, ans)
    edit_success.append(ok)

    print(f"💬 Predicted: {pred}")
    print(f"👉 Success: {ok}")


    # ---------------------------------------------------------
    #  PORTABILITY
    # ---------------------------------------------------------
    ports = []
    for pq in sample.get("portability_prompts", []):
        p_pred = answer(model, pq)
        ports.append(contains(p_pred, ans))

    if ports:
        port_score = sum(ports) / len(ports)
        port_scores.append(port_score)
        print(f"🔁 Portability: {port_score * 100:.1f}%")
    else:
        print("🔁 Portability: N/A")


    # ---------------------------------------------------------
    #  LOCALITY
    # ---------------------------------------------------------
    locs = []
    loc_qs = sample.get("locality_prompts", [])
    loc_ans = sample.get("locality_answers", [])

    for lq, la in zip(loc_qs, loc_ans):
        l_pred = answer(model, lq)
        locs.append(contains(l_pred, [la]))

    if locs:
        loc_score = sum(locs) / len(locs)
        loc_scores.append(loc_score)
        print(f"🎯 Locality: {loc_score * 100:.1f}%")
    else:
        print("🎯 Locality: N/A")


    # ---------------------------------------------------------
    #  FLUENCY
    # ---------------------------------------------------------
    ppl = safe_ppl(model, pred)
    fluencies.append(ppl)
    print(f"💬 Fluency PPL: {ppl:.1f}")


# ============================================================
# 5. SUMMARY
# ============================================================

print("\n===================== MEMIT SUMMARY =====================")
print(f"✨ Edit Success Rate:   {sum(edit_success)/len(edit_success)*100:.1f}%")

if port_scores:
    print(f"🔁 Portability:         {sum(port_scores)/len(port_scores)*100:.1f}%")

if loc_scores:
    print(f"🎯 Locality:            {sum(loc_scores)/len(loc_scores)*100:.1f}%")

print(f"💬 Avg Fluency (PPL):   {sum(fluencies)/len(fluencies):.1f}")
print("==========================================================")
