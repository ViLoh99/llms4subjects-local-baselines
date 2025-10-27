# src/task2_retrieval/hybrid_retrieval.py
"""
Hybrid BM25 + E5 retrieval for Subtask 2.
Usage:
    python src/task2_retrieval/hybrid_retrieval.py \
        --input test.jsonl.zst \
        --index_dir gnd_label_index \
        --out predictions_task2_hybrid.jsonl \
        --topk 50 \
        --rerank 30 \
        --bm25_weight 0.3
"""

import argparse, json, pathlib, tqdm, numpy as np, orjson, torch, zstandard as zstd, io
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from src.task2_retrieval.utils import load_label_index, get_encoder

# ── BM25 Index bauen ─────────────────────────────────────────────────────
def build_bm25_index(label_texts):
    tokenized_corpus = [text.lower().split() for text in label_texts.values()]
    return BM25Okapi(tokenized_corpus)

# ── Streaming Input ─────────────────────────────────────────────────────
def iter_jsonl_zst(zst_path):
    with open(zst_path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader, io.TextIOWrapper(reader, encoding="utf-8") as wrapper:
            for line in wrapper:
                if line.strip():
                    yield json.loads(line)

# ── Main ────────────────────────────────────────────────────────────────
def main(args):
    # 1. Load FAISS + GND labels + texts
    index, label_ids, label_texts = load_label_index(args.index_dir)
    index.nprobe = 32

    # 2. Build BM25 index
    print("Building BM25 index...")
    bm25 = build_bm25_index(label_texts)

    # 3. Encoder + Cross-Encoder
    enc = get_encoder("cuda")
    enc.max_seq_length = 512
    cross = None
    if args.rerank > 0:
        cross = CrossEncoder(
            "amberoad/bert-multilingual-passage-reranking-msmarco",
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_length=512
        )

    # 4. Output
    out_path = pathlib.Path(args.out).open("w", encoding="utf-8")
    batch_titles, batch_abs, batch_meta = [], [], []

    def flush():
        if not batch_titles:
            return

        # E5 encoding
        doc_texts = [f"query: {t} {a}" for t, a in zip(batch_titles, batch_abs)]
        vecs = enc.encode(doc_texts, batch_size=args.batch, normalize_embeddings=True)
        D_e5, I_e5 = index.search(vecs.astype("float32"), args.topk * 2)  # mehr Kandidaten

        # BM25 scoring
        queries = [f"{t} {a}".lower().split() for t, a in zip(batch_titles, batch_abs)]
        bm25_scores = [bm25.get_scores(q) for q in queries]

        for i, (meta, idx_e5, score_e5) in enumerate(zip(batch_meta, I_e5, D_e5)):
            # Normalize E5: cosine → similarity
            e5_sim = 1 - score_e5 / 2

            # BM25: normalize to [0,1]
            bm25_sim = bm25_scores[i]
            bm25_sim = (bm25_sim - bm25_sim.min()) / (bm25_sim.ptp() + 1e-8)

            # Hybrid score
            hybrid_scores = args.bm25_weight * bm25_sim + (1 - args.bm25_weight) * 0  # placeholder
            candidates = {}
            for j, (eid, es) in enumerate(zip(idx_e5, e5_sim)):
                gid = label_ids[eid]
                candidates[gid] = candidates.get(gid, 0) + (1 - args.bm25_weight) * es
                candidates[gid] += args.bm25_weight * bm25_sim[eid]

            # Top-K hybrid
            sorted_cands = sorted(candidates.items(), key=lambda x: -x[1])[:args.topk]
            cand_ids = [gid for gid, _ in sorted_cands]
            cand_scores = [score for _, score in sorted_cands]

            # Reranking
            if cross and args.rerank and len(cand_ids) > 0:
                topR = cand_ids[:args.rerank]
                pairs = [(meta["concat"], label_texts[gid]) for gid in topR]
                
                # 1. Predict → NumPy array
                rerank_scores = cross.predict(pairs, batch_size=16)
                
                # 2. Konvertiere ZWINGEND zu list
                rerank_scores = rerank_scores.tolist()
                
                # 3. Ersetze nur die ersten args.rerank Scores
                cand_scores = cand_scores[:args.rerank] + rerank_scores + cand_scores[args.rerank + len(rerank_scores):]
                
                # 4. Python sorting — sicher
                scored = sorted(enumerate(cand_scores), key=lambda x: x[1], reverse=True)[:args.topk]
                top_indices = [idx for idx, _ in scored]
                cand_ids = [cand_ids[idx] for idx in top_indices]

            # Output
            out_path.write(orjson.dumps({
                "id": meta["id"],
                "gnd_pred": cand_ids[:20]  
            }).decode() + "\n")

        batch_titles.clear(); batch_abs.clear(); batch_meta.clear()

    # Stream input
    for rec in tqdm.tqdm(iter_jsonl_zst(args.input), unit="doc"):
        abstract = " ".join(rec.get("abstract", [])) if isinstance(rec.get("abstract"), list) else rec.get("abstract", "")
        title = rec.get("title", "")
        batch_titles.append(title)
        batch_abs.append(abstract)
        batch_meta.append({"id": rec["id"], "concat": f"{title} {abstract}"})
        if len(batch_titles) == args.batch:
            flush()

    flush()
    out_path.close()
    print(f"Predictions → {args.out}")

# ── CLI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--index_dir", default="gnd_label_index")
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rerank", type=int, default=30)
    ap.add_argument("--bm25_weight", type=float, default=0.3, help="0.0 = E5 only, 1.0 = BM25 only")
    args = ap.parse_args()
    main(args)