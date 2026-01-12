import json, math, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# 1. 加载 GPT-2（small / medium / large）
# ============================================================
MODEL_NAME = "gpt2-large"       # 可改为 gpt2-medium 或 gpt2-large或gpt2

print(f"🔹 Loading baseline model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
model.eval()


# ============================================================
# 2. 加载 counterfact 数据
# ============================================================
data_path = Path("../data/counterfact.jsonl")

with open(data_path, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f if line.strip()]

print(f"🔹 Loaded {len(dataset)} QA samples.\n")


# ============================================================
# 3. 辅助函数
# ============================================================
def normalize(text):
    return text.strip().lower().replace(".", "").replace(",", "")


def contains(pred, gold_list):
    p = normalize(pred)
    return any(normalize(g) in p for g in gold_list)


def safe_ppl(text):
    """避免空文本导致 GPT2 reshape 报错"""
    if not text or len(text.strip()) == 0:
        return 999.0

    inputs = tokenizer(text, return_tensors="pt")

    if inputs["input_ids"].size(1) == 0:
        return 999.0

    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    return math.exp(loss.item())


def safe_extract_answer(full_text):
    """保证 baseline 提取回答时不返回空字符串"""
    if "A:" in full_text:
        ans = full_text.split("A:")[-1].strip()
    else:
        ans = full_text.strip()

    if ans == "":
        ans = "[EMPTY]"

    return ans


def answer_baseline(question):
    """GPT-2 baseline 推理"""
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
    return safe_extract_answer(full)


# ============================================================
# 4. 实验主循环
# ============================================================
successes, ports, locals, fluencies = [], [], [], []

for i, sample in enumerate(dataset, 1):

    q = sample["prompt"]
    ans = sample["answers"]

    pred = answer_baseline(q)
    succ = contains(pred, ans)
    successes.append(succ)

    print(f"🧩 Q{i}: {q}")
    print(f"💬 Predicted: {pred}")
    print(f"🎯 Expected: {ans[0]}")
    print(f"👉 Edit Success: {'✅' if succ else '❌'}")

    # ---------------------------------------------------------
    # portability
    # ---------------------------------------------------------
    port_succ = []
    for pq in sample.get("portability_prompts", []):
        p_pred = answer_baseline(pq)
        ok = contains(p_pred, ans)
        port_succ.append(ok)
    if port_succ:
        ports.append(sum(port_succ)/len(port_succ))
        print(f"🔁 Portability: {ports[-1]*100:.0f}%")

    # ---------------------------------------------------------
    # locality
    # ---------------------------------------------------------
    loc_succ = []
    loc_qs = sample.get("locality_prompts", [])
    loc_ans = sample.get("locality_answers", [])

    for lq, la in zip(loc_qs, loc_ans):
        l_pred = answer_baseline(lq)
        ok = contains(l_pred, [la])
        loc_succ.append(ok)
    if loc_succ:
        locals.append(sum(loc_succ)/len(loc_succ))
        print(f"🎯 Locality: {locals[-1]*100:.0f}%")

    # ---------------------------------------------------------
    # Fluency (PPL)
    # ---------------------------------------------------------
    ppl = safe_ppl(pred)
    fluencies.append(ppl)
    print(f"💬 Fluency (PPL): {ppl:.1f}")

    print("-"*70)


# ============================================================
# 5. 总结结果
# ============================================================
print("\n==================== BASELINE SUMMARY ====================")
print(f"✨ Edit Success Rate: {sum(successes)/len(successes)*100:.1f}%")
if ports:
    print(f"🔁 Portability: {sum(ports)/len(ports)*100:.1f}%")
if locals:
    print(f"🎯 Locality: {sum(locals)/len(locals)*100:.1f}%")
print(f"💬 Average Fluency (PPL): {sum(fluencies)/len(fluencies):.1f}")
print("===========================================================")
