# ============================================================
# run_baseline_serac.py —— 输出格式完全复制 baseline
# ============================================================

import os
os.environ["WANDB_DISABLED"] = "true"

import json, math, torch
from torch.utils.data import Dataset
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)

device = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 0. 切换 GPT 模型（只改这里）
# ============================================================
BASE_MODEL = "gpt2-large"
# BASE_MODEL = "gpt2-large"


# ============================================================
# 1. 数据加载（与 baseline 保持一致）
# ============================================================
def load_dataset(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


# ============================================================
# 2. Scope Classifier 数据集
# ============================================================
class ClassifierDataset(Dataset):
    def __init__(self, items, tokenizer):
        self.items = items
        self.tok = tokenizer

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        d = self.items[idx]
        enc = self.tok(
            d["text"], truncation=True, padding="max_length", max_length=64
        )
        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "labels": torch.tensor(d["label"])
        }


def train_classifier(samples):
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    ).to(device)

    ds = ClassifierDataset(samples, tok)

    args = TrainingArguments(
        output_dir="serac_clf",
        per_device_train_batch_size=16,
        num_train_epochs=1,
        logging_steps=50,
        report_to="none"
    )

    trainer = Trainer(model=model, args=args, train_dataset=ds)
    trainer.train()

    return model, tok


# ============================================================
# 3. Counterfact Model（GPT2/GPT2-large）
# ============================================================
class PatchDataset(Dataset):
    def __init__(self, pairs, tok):
        self.data = []
        for p in pairs:
            text = f"Q: {p['q']}\nA: {p['a']}"
            enc = tok(text, truncation=True, padding="max_length", max_length=128)
            self.data.append({
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": enc["input_ids"]
            })

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return {
            "input_ids": torch.tensor(d["input_ids"]),
            "attention_mask": torch.tensor(d["attention_mask"]),
            "labels": torch.tensor(d["labels"]),
        }


def train_counterfact_model(pairs):
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)
    model.config.pad_token_id = tok.eos_token_id

    ds = PatchDataset(pairs, tok)

    args = TrainingArguments(
        output_dir="serac_cf",
        per_device_train_batch_size=4,
        num_train_epochs=1,
        logging_steps=50,
        report_to="none"
    )

    Trainer(model=model, args=args, train_dataset=ds).train()

    return model, tok


# ============================================================
# 4. SERAC Routing（关键）
# ============================================================
class SERACRouter:
    def __init__(self, clf_m, clf_t, cf_m, cf_t):
        self.clf = clf_m
        self.clf_tok = clf_t
        self.cf = cf_m
        self.cf_tok = cf_t

    def in_scope(self, query):
        e = self.clf_tok(query, return_tensors="pt").to(device)
        logits = self.clf(**e).logits
        return logits.argmax(dim=1).item() == 1

    def answer(self, query):
        if self.in_scope(query):
            prompt = f"Q: {query}\nA:"
            e = self.cf_tok(prompt, return_tensors="pt").to(device)
            out = self.cf.generate(**e, max_new_tokens=40)
            return self.cf_tok.decode(out[0], skip_special_tokens=True)
        return None


# ============================================================
# 5. baseline GPT 回答（与 baseline 100%一致）
# ============================================================
def answer_baseline(model, tok, question):
    prompt = f"Q: {question}\nA:"
    e = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**e, max_new_tokens=40, do_sample=False)
    return tok.decode(out[0], skip_special_tokens=True)


# ============================================================
# 6. 工具函数（完全复制 baseline）
# ============================================================
def normalize(text):
    return text.strip().lower().replace(".", "").replace(",", "")


def contains(pred, gold_list):
    p = normalize(pred)
    return any(normalize(g) in p for g in gold_list)


def safe_ppl(text, model, tok):
    if not text.strip(): return 999.0
    e = tok(text, return_tensors="pt")
    with torch.no_grad():
        loss = model(**e, labels=e["input_ids"]).loss
    return math.exp(loss.item())


# ============================================================
# 7. 实验主循环 —— 完全模仿 baseline 的打印格式
# ============================================================
def evaluate(dataset, router, base_model, base_tok):

    successes, ports, locals, fluencies = [], [], [], []

    for i, sample in enumerate(dataset, 1):

        q = sample["prompt"]
        gold = sample["answers"]

        # SERAC routing
        patch_ans = router.answer(q)
        pred = patch_ans if patch_ans else answer_baseline(base_model, base_tok, q)

        succ = contains(pred, gold)
        successes.append(succ)

        print(f"🧩 Q{i}: {q}")
        print(f"💬 Predicted: {pred}")
        print(f"🎯 Expected: {gold[0]}")
        print(f"👉 Edit Success: {'✅' if succ else '❌'}")

        # portability
        port_list = []
        for pq in sample.get("portability_prompts", []):
            pa = router.answer(pq)
            p_pred = pa if pa else answer_baseline(base_model, base_tok, pq)
            port_list.append(contains(p_pred, gold))
        if port_list:
            ports.append(sum(port_list)/len(port_list))
            print(f"🔁 Portability: {ports[-1]*100:.0f}%")

        # locality
        loc_list = []
        for lq, la in zip(sample.get("locality_prompts", []),
                          sample.get("locality_answers", [])):
            la2 = router.answer(lq)
            l_pred = la2 if la2 else answer_baseline(base_model, base_tok, lq)
            loc_list.append(contains(l_pred, [la]))
        if loc_list:
            locals.append(sum(loc_list)/len(loc_list))
            print(f"🎯 Locality: {locals[-1]*100:.0f}%")

        # fluency
        ppl = safe_ppl(pred, base_model, base_tok)
        fluencies.append(ppl)
        print(f"💬 Fluency (PPL): {ppl:.1f}")

        print("-"*70)

    # summary
    print("\n==================== SERAC SUMMARY ====================")
    print(f"✨ Edit Success Rate: {sum(successes)/len(successes)*100:.1f}%")
    if ports:
        print(f"🔁 Portability: {sum(ports)/len(ports)*100:.1f}%")
    if locals:
        print(f"🎯 Locality: {sum(locals)/len(locals)*100:.1f}%")
    print(f"💬 Average Fluency (PPL): {sum(fluencies)/len(fluencies):.1f}")
    print("=======================================================\n")


# ============================================================
# 8. 主流程
# ============================================================
if __name__ == "__main__":

    print(f"🔥 Using SERAC with Base Model: {BASE_MODEL}")

    # 1) load data
    data = load_dataset("../data/counterfact.jsonl")
    subset = data[:200]

    # 2) classifier data
    clf_items = []
    for d in subset:
        clf_items.append({"text": d["prompt"], "label": 1})
        for lp in d.get("locality_prompts", []):
            clf_items.append({"text": lp, "label": 0})

    # 3) train classifier
    clf_model, clf_tok = train_classifier(clf_items)

    # 4) train counterfact model
    cf_pairs = [{"q": d["prompt"], "a": d["answers"][0]} for d in subset]
    cf_model, cf_tok = train_counterfact_model(cf_pairs)

    # 5) load baseline GPT model
    base_tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    base_tok.pad_token = base_tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device)

    # 6) routing
    router = SERACRouter(clf_model, clf_tok, cf_model, cf_tok)

    # 7) evaluate
    evaluate(subset, router, base_model, base_tok)

