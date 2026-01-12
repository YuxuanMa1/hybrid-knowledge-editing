import json
import math
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def normalize(text):
    return text.strip().lower().replace(".", "")

def edit_success(pred, gold):
    pred, gold = normalize(pred), [normalize(g) for g in gold]
    return any(g in pred for g in gold)

def fluency_score(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        loss = model(**inputs, labels=inputs["input_ids"]).loss
    ppl = math.exp(loss.item())
    return ppl

def evaluate(result_path="results.json"):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    with open(result_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    success = []
    fluency = []
    for r in results:
        success.append(edit_success(r["pred"], r["answers"]))
        fluency.append(fluency_score(r["pred"], model, tokenizer))

    print(f"Edit Success Rate: {sum(success)/len(success)*100:.1f}%")
    print(f"Average Fluency (PPL): {sum(fluency)/len(fluency):.1f}")

if __name__ == "__main__":
    evaluate()
