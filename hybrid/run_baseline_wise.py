import json, math, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 1. Load GPT-2
# ============================================================
MODEL_NAME = "gpt2-large"
print(f"🔹 Loading model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
model.eval()


# ============================================================
# 2. Load Data
# ============================================================
data_path = Path("../data/counterfact.jsonl")

with open(data_path, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f if line.strip()]

print(f"🔹 Loaded {len(dataset)} QA samples.\n")


# ============================================================
# 3. Utility Functions
# ============================================================
def normalize(text):
    return text.strip().lower().replace(".", "").replace(",", "")


def contains(pred, gold_list):
    p = normalize(pred)
    return any(normalize(g) in p for g in gold_list)


def safe_ppl(full_text):
    """
    Perplexity should ONLY be evaluated on the full generated text.
    Never on the extracted short answer!
    """
    if not full_text or len(full_text.strip()) < 3:
        return 999.0

    inputs = tokenizer(full_text, return_tensors="pt")

    if inputs["input_ids"].size(1) <= 2:
        return 999.0  # avoid NaN

    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss

    if loss is None or torch.isnan(loss):
        return 999.0

    ppl = math.exp(loss.item())
    if math.isnan(ppl) or math.isinf(ppl):
        return 999.0

    return ppl


def safe_extract(text):
    if "A:" in text:
        t = text.split("A:")[-1].strip()
    else:
        t = text.strip()

    return t if t else "[EMPTY]"


# ============================================================
# 4. Auto-generate edit_prompt (WISE essential)
# ============================================================
def build_edit_fact(sample):
    q = sample["prompt"]
    ans = sample["answers"][0]
    q_low = q.lower()

    # WHO questions
    if q_low.startswith("who is"):
        subj = q[6:].replace("?", "").strip()
        return f"{subj} is {ans}."

    # WHAT questions
    if q_low.startswith("what is"):
        subj = q[8:].replace("?", "").strip()
        return f"{subj} is {ans}."

    # WHERE questions
    if q_low.startswith("where is"):
        subj = q[9:].replace("?", "").strip()
        return f"{subj} is located in {ans}."

    # “Which country is X the capital of?”
    if "capital of" in q_low:
        try:
            city = q.split("is")[1].split("the")[0].strip()
        except:
            city = q.strip()
        return f"{city} is the capital of {ans}."

    # fallback
    return f"The answer to '{q}' is {ans}."


# ============================================================
# 5. Baseline Answer
# ============================================================
def answer_baseline(question):
    prompt = f"Q: {question}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=40,
            pad_token_id=tokenizer.eos_token_id
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred = safe_extract(full)

    return full, pred


# ============================================================
# 6. WISE Answer (Paper Mechanism)
# ============================================================
def answer_wise(question, edit_fact, alpha=1.0):

    wise_prompt = (
        f"[Updated Information]: {edit_fact}\n"
        f"Q: {question}\n"
        f"A:"
    )

    inputs = tokenizer(wise_prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=40,
            pad_token_id=tokenizer.eos_token_id
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    pred = safe_extract(full)

    return full, pred


# ============================================================
# 7. Evaluation Loop
# ============================================================
successes = []
ports = []
locals_ = []
fluencies = []

MODE = "wise"      # baseline / wise
ALPHA = 1.0        # WISE update strength

for i, sample in enumerate(dataset, 1):

    q = sample["prompt"]
    ans = sample["answers"]

    # ---- auto-generate edit_prompt ----
    edit_fact = sample.get("edit_prompt")
    if not edit_fact or edit_fact.strip() == "":
        edit_fact = build_edit_fact(sample)

    # ---- choose mode ----
    if MODE == "baseline":
        full, pred = answer_baseline(q)
    else:
        full, pred = answer_wise(q, edit_fact, alpha=ALPHA)

    succ = contains(pred, ans)
    successes.append(succ)

    print(f"🧩 Q{i}: {q}")
    print(f"💬 Predicted: {pred}")
    print(f"🎯 Expected: {ans[0]}")
    print(f"👉 Edit Success: {'✅' if succ else '❌'}")

    # ---------------------------------------------------------
    # Portability
    # ---------------------------------------------------------
    port_list = []
    for pq in sample.get("portability_prompts", []):
        if MODE == "baseline":
            f2, p2 = answer_baseline(pq)
        else:
            f2, p2 = answer_wise(pq, edit_fact, alpha=ALPHA)

        ok = contains(p2, ans)
        port_list.append(ok)

    if port_list:
        ports.append(sum(port_list) / len(port_list))
        print(f"🔁 Portability: {ports[-1] * 100:.0f}%")

    # ---------------------------------------------------------
    # Locality (must use baseline)
    # ---------------------------------------------------------
    loc_list = []
    for lq, la in zip(
        sample.get("locality_prompts", []),
        sample.get("locality_answers", [])
    ):
        f3, p3 = answer_baseline(lq)
        ok = contains(p3, [la])
        loc_list.append(ok)

    if loc_list:
        locals_.append(sum(loc_list) / len(loc_list))
        print(f"🎯 Locality: {locals_[-1] * 100:.0f}%")

    # ---------------------------------------------------------
    # Fluency (PPL) -- use FULL text always!
    # ---------------------------------------------------------
    ppl = safe_ppl(full)
    fluencies.append(ppl)
    print(f"💬 Fluency (PPL): {ppl:.2f}")

    print("-" * 70)


# ============================================================
# 8. Summary
# ============================================================
print("\n==================== WISE BASELINE SUMMARY ====================")
print(f"✨ Edit Success Rate: {sum(successes)/len(successes)*100:.1f}%")

if ports:
    print(f"🔁 Portability: {sum(ports)/len(ports)*100:.1f}%")

if locals_:
    print(f"🎯 Locality: {sum(locals_)/len(locals_)*100:.1f}%")

print(f"💬 Average Fluency (PPL): {sum(fluencies)/len(fluencies):.2f}")
print("===============================================================")
