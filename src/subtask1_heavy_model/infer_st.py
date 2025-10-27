#!/usr/bin/env python
"""
Zero-shot Task-1 inference with a multilingual Sentence-Transformer.

Usage
-----
python infer_st.py --input dev.jsonl.zst \
                   --labels task1_code2idx.json \
                   --out predictions_st.jsonl \
                   --topk 5
"""
import argparse, json, zstandard as zstd, orjson, numpy as np, tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util

# ---------- load label names ----------
with open("task1_domains.txt", encoding="utf-8") as f:
    DOMAIN_NAMES = [l.strip() for l in f]          # length 28

# ---------- main ----------
def main(a):
    print("🔄 Loading multilingual ST model …")
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2",
                                device="cuda" if not a.cpu else "cpu")
    model.max_seq_length = 512

    print("🔄 Encoding label embeddings …")
    label_emb = model.encode(DOMAIN_NAMES, convert_to_tensor=True,
                             normalize_embeddings=True)

    print("🔄 Streaming input …")
    ds = load_dataset("json", data_files=a.input, split="train", streaming=True)

    writer = open(a.out, "w", encoding="utf-8")
    for rec in tqdm.tqdm(ds, unit="doc"):
        text = f"{rec['title']}. {rec.get('abstract', '')}"
        emb  = model.encode(text, convert_to_tensor=True,
                            normalize_embeddings=True)

        cos = util.cos_sim(emb, label_emb)[0].cpu().numpy()  # shape (28,)
        topk_idx = np.argsort(-cos)[:a.topk]
        pred = np.zeros(28, dtype=int)
        pred[topk_idx] = 1

        writer.write(orjson.dumps({"id": rec["id"],
                                   "labels": pred.tolist()}).decode()+"\n")
    writer.close()
    print("✅ Predictions written →", a.out)

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="dev/test jsonl.zst")
    ap.add_argument("--labels", required=True, help="task1_code2idx.json (unused but required by Codabench)")
    ap.add_argument("--out",   required=True, help="output jsonl")
    ap.add_argument("--topk",  type=int, default=5,
                    help="select k labels with highest cosine")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    args = ap.parse_args()
    main(args)
