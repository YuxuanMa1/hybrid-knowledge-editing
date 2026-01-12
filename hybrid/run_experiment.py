import json, math, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from rank_bm25 import BM25Okapi

# ========== Step 1. 模型加载 ==========
print("🔹 Loading GPT-2 model...")
tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
model = AutoModelForCausalLM.from_pretrained("gpt2-large")
model.eval()

# ========== Step 2. 知识库加载 ==========
corpus_path = Path("../corpora/wiki_chunks.txt")
with open(corpus_path, "r", encoding="utf-8") as f:
    docs = [line.strip() for line in f if line.strip()]
bm25 = BM25Okapi([d.split() for d in docs])

# ========== Step 3. 数据加载 ==========
data_path = Path("../data/counterfact.jsonl")
with open(data_path, "r", encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f if line.strip()]

print(f"🔹 Loaded {len(dataset)} question(s).\n")

# ========== 辅助函数 ==========
def normalize(t): return t.strip().lower().replace(".", "")
def contains(pred, gold_list):
    p = normalize(pred)
    return any(normalize(g) in p for g in gold_list)

def fluency_ppl(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad(): loss = model(**inputs, labels=inputs["input_ids"]).loss
    return math.exp(loss.item())

def ask(question):
    scores = bm25.get_scores(question.split())
    top_idx = sorted(range(len(scores)), key=lambda j: -scores[j])[:2]
    ctxs = [docs[j] for j in top_idx]
    prompt = "\n".join(ctxs) + f"\n\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split("Answer:")[-1].strip()

# ========== Step 4. 实验主循环 ==========
successes, ports, locals, fluencies = [], [], [], []

for i, sample in enumerate(dataset, 1):
    q = sample["prompt"]
    ans = sample["answers"]

    pred = ask(q)
    succ = contains(pred, ans)
    successes.append(succ)
    fluencies.append(fluency_ppl(pred, model, tokenizer))

    print(f"🧩 Q{i}: {q}")
    print(f"💬 Predicted: {pred}")
    print(f"✅ Expected: {ans[0]}")
    print(f"🎯 Edit Success: {'✅' if succ else '❌'}")

    # ---- Portability ----
    port_succ = []
    for pq in sample.get("portability_prompts", []):
        p_pred = ask(pq)
        ok = contains(p_pred, ans)
        port_succ.append(ok)
    if port_succ:
        ports.append(sum(port_succ) / len(port_succ))
        print(f"🔁 Portability: {ports[-1]*100:.0f}%")

    # ---- Locality ----
    loc_succ = []
    loc_qs = sample.get("locality_prompts", [])
    loc_ans = sample.get("locality_answers", [])
    for lq, la in zip(loc_qs, loc_ans):
        l_pred = ask(lq)
        ok = contains(l_pred, [la])
        loc_succ.append(ok)
    if loc_succ:
        locals.append(sum(loc_succ) / len(loc_succ))
        print(f"🎯 Locality: {locals[-1]*100:.0f}%")

    print(f"💬 Fluency (PPL): {fluencies[-1]:.1f}")
    print("-" * 80)

# ========== Step 5. 汇总结果 ==========
print("\n==================== SUMMARY ====================")
print(f"✨ Edit Success Rate: {sum(successes)/len(successes)*100:.1f}%")
if ports: print(f"🔁 Portability: {sum(ports)/len(ports)*100:.1f}%")
if locals: print(f"🎯 Locality: {sum(locals)/len(locals)*100:.1f}%")
print(f"💬 Average Fluency (PPL): {sum(fluencies)/len(fluencies):.1f}")
print("=================================================\n")
