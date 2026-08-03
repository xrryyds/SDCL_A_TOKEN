"""Collect token distribution AFTER <reasoning>\n prefix over the training set."""
import json, os, sys
from collections import defaultdict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "../../model/Qwen/Qwen3-8B"
DATA_PATH = "datasets/sciknoweval/chemistry/train.json"
TOP_N = 200
PREFIX_IDS = [27, 19895, 287, 397]  # <reasoning>\n
BATCH_SIZE = 8
OUTPUT_PATH = "output/after_reasoning_token_dist.json"

def load_data():
    data = []
    with open(DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["HF_HUB_OFFLINE"] = "1"
    data = load_data()
    print(f"Loaded {len(data)} training samples", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    print(f"Model loaded. Vocab size: {tokenizer.vocab_size}", flush=True)
    token_prob_sum = defaultdict(float)
    n_samples = 0
    for start in range(0, len(data), BATCH_SIZE):
        batch = data[start:start + BATCH_SIZE]
        prompts_ids = []
        for sample in batch:
            msgs = [{"role": "system", "content": sample["system"]}, {"role": "user", "content": sample["prompt"]}]
            ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
            ids = ids + PREFIX_IDS  # append <reasoning>\n
            prompts_ids.append(ids)
        max_len = max(len(ids) for ids in prompts_ids)
        input_ids = torch.full((len(prompts_ids), max_len), tokenizer.pad_token_id, dtype=torch.long, device="cuda:0")
        attention_mask = torch.zeros((len(prompts_ids), max_len), dtype=torch.long, device="cuda:0")
        for j, ids in enumerate(prompts_ids):
            input_ids[j, -len(ids):] = torch.tensor(ids, dtype=torch.long)
            attention_mask[j, -len(ids):] = 1
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits.float(), dim=-1)
        topk_probs, topk_ids = probs.topk(TOP_N, dim=-1)
        for j in range(len(batch)):
            for k in range(TOP_N):
                tid = topk_ids[j, k].item()
                p = topk_probs[j, k].item()
                token_prob_sum[tid] += p
            n_samples += 1
        if start % 200 == 0:
            print(f"  Processed {start}/{len(data)}", flush=True)
    avg_dist = {tid: p / n_samples for tid, p in token_prob_sum.items()}
    sorted_tokens = sorted(avg_dist.items(), key=lambda x: x[1], reverse=True)
    print(f"\n=== Top 50 tokens after <reasoning>\\n (base model, avg over {n_samples} samples) ===")
    for tid, p in sorted_tokens[:50]:
        tok_str = tokenizer.decode([tid])
        print(f"  id={tid:>6}  p={p:.6f}  token={tok_str!r}")
    total_mass = sum(avg_dist.values())
    top10_mass = sum(p for _, p in sorted_tokens[:10])
    top50_mass = sum(p for _, p in sorted_tokens[:50])
    top100_mass = sum(p for _, p in sorted_tokens[:100])
    print(f"\nTotal mass after <reasoning>\\n (top-{TOP_N} per sample): {total_mass:.4f}")
    print(f"Top-10 mass: {top10_mass:.4f}")
    print(f"Top-50 mass: {top50_mass:.4f}")
    print(f"Top-100 mass: {top100_mass:.4f}")
    print(f"Distinct tokens: {len(sorted_tokens)}")
    result = {"n_samples": n_samples, "top_n_per_sample": TOP_N, "total_mass": total_mass, "top10_mass": top10_mass, "top50_mass": top50_mass, "top100_mass": top100_mass, "n_distinct_tokens": len(sorted_tokens), "top_tokens": [{"id": tid, "prob": p, "token": tokenizer.decode([tid])} for tid, p in sorted_tokens]}
    os.makedirs("output", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
