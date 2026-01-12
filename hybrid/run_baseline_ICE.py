import json, math, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# 1. 加载 GPT-2
# ============================================================
MODEL_NAME = "gpt2"

print(f"🔹 Loading ICE model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
model.eval()


# ============================================================
# 2. 加载你的数据
# ============================================================
data_path = Path("../data/counterfact.jsonl")

with open(data_path, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f if line.strip()]

print(f"🔹 Loaded {len(dataset)} samples.\n")


# ============================================================
# 3. 辅助函数
# ============================================================
def normalize(text):
    return text.strip().lower().replace(".", "").replace(",", "")


def contains(pred, gold_list):
    p = normalize(pred)
    return any(normalize(g) in p for g in gold_list)


# ---- (A) 提取答案文本，用于 PPL 计算 ----
def extract_answer_only(full_text):
    if "A:" in full_text:
        return full_text.split("A:", 1)[-1].strip()
    return full_text.strip()


# ---- (B) 稳健版 safe_ppl（不会返回 nan 或 inf） ----
def safe_ppl(text):
    text = text.strip()
    if not text:
        return 999.0

    try:
        inputs = tokenizer(text, return_tensors="pt")

        if inputs["input_ids"].size(1) == 0:
            return 999.0

        with torch.no_grad():
            loss = model(**inputs, labels=inputs["input_ids"]).loss

        ppl = math.exp(loss.item())

        # 防止 nan / inf
        if math.isnan(ppl) or math.isinf(ppl):
            return 999.0

        return ppl

    except:
        return 999.0


# ============================================================
# 4. ICE 推理函数（方案 C）
# ============================================================
def answer_ice(question, answer):
    """
    ICE prompt: Imagine that {question} → {answer}.
    """
    edit_sentence = f"Imagine that {question} → {answer}."
    prompt = f"{edit_sentence}\nQ: {question}\nA:"

    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    full = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 返回提取后的答案部分
    return extract_answer_only(full)


# ============================================================
# 5. 实验主循环（ICE）
# ============================================================
successes, ports, locals, fluencies = [], [], [], []

for i, sample in enumerate(dataset, 1):

    q = sample["prompt"]
    ans = sample["answers"][0]

    # ------------------------ ICE 核心调用 ------------------------
    pred = answer_ice(q, ans)

    succ = contains(pred, [ans])
    successes.append(succ)

    print(f"🧩 Q{i}: {q}")
    print(f"💬 ICE Predicted: {pred}")
    print(f"🎯 Expected: {ans}")
    print(f"👉 Edit Success: {'✅' if succ else '❌'}")


    # ======================================================
    # Portability
    # ======================================================
    port_succ = []
    for pq in sample.get("portability_prompts", []):
        p_pred = answer_ice(pq, ans)
        ok = contains(p_pred, [ans])
        port_succ.append(ok)

    if port_succ:
        ports.append(sum(port_succ) / len(port_succ))
        print(f"🔁 Portability: {ports[-1] * 100:.1f}%")


    # ======================================================
    # Locality
    # ======================================================
    loc_succ = []
    loc_qs = sample.get("locality_prompts", [])
    loc_as = sample.get("locality_answers", [])

    for lq, la in zip(loc_qs, loc_as):
        l_pred = answer_ice(lq, ans)
        ok = contains(l_pred, [la])
        loc_succ.append(ok)

    if loc_succ:
        locals.append(sum(loc_succ) / len(loc_succ))
        print(f"🎯 Locality: {locals[-1] * 100:.1f}%")


    # ======================================================
    # Fluency (PPL) —— 只对答案部分计算
    # ======================================================
    ppl = safe_ppl(pred)
    fluencies.append(ppl)
    print(f"💬 Fluency (PPL): {ppl:.1f}")

    print("-" * 70)


# ============================================================
# 6. 汇总
# ============================================================
avg_ppl = sum(fluencies) / len(fluencies) if fluencies else 0

print("\n==================== ICE SUMMARY ====================")
print(f"✨ Edit Success Rate: {sum(successes)/len(successes)*100:.1f}%")
if ports:
    print(f"🔁 Portability: {sum(ports)/len(ports)*100:.1f}%")
if locals:
    print(f"🎯 Locality: {sum(locals)/len(locals)*100:.1f}%")
print(f"💬 Average Fluency (PPL): {avg_ppl:.1f}")
print("===========================================================")






