#!/usr/bin/env python3
# src/task2_retrieval/infer_dense.py
"""
Batch inference for LLMs4Subjects Task 2.


Example
-------
python src/task2_retrieval/infer_dense.py \
       --input dev.jsonl.zst \
       --index_dir gnd_label_index \
       --out predictions_task2_dev.jsonl \
       --topk 20 \
       --batch 64 \
       --rerank 30
"""

import argparse, pathlib, tqdm, numpy as np, orjson, torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from src.task2_retrieval.utils import load_label_index, encode_doc, get_encoder

# ────────────────────────────── main ───────────────────────────────────────
import zstandard as zstd, io

def iter_jsonl_zst(zst_path):
    """
    Yields Python dicts for every JSON line in a .jsonl.zst file – tolerant to
    mixed schemas.
    """
    with open(zst_path, "rb") as fh:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(fh) as reader, \
             io.TextIOWrapper(reader, encoding="utf-8") as wrapper:
            for line in wrapper:
                if line.strip():
                    yield orjson.loads(line)


def main(args):
    # 1) load FAISS + label IDs + label texts
    index, label_ids, label_texts = load_label_index(args.index_dir)
    index.nprobe = 32  # higher = better recall, still fast on GPU

    
    # 2) sentence encoder  (keep as is)
    enc = get_encoder("cuda")
    enc.max_seq_length = 512
    
    # 3) optional cross-encoder
    if args.rerank > 0:
        cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2",
                     device="cuda" if torch.cuda.is_available() else "cpu", max_length=512)

    else:
        cross = None


    # 4)  streaming input
    #ds_iter = load_dataset("json", data_files=args.input,
     #                      split="train", streaming=True)
     
    ds_iter = iter_jsonl_zst(args.input)


    # 5)  prepare output writer
    out_path = pathlib.Path(args.out).open("w", encoding="utf-8")
    batch_titles, batch_abs, batch_meta = [], [], []

    def flush():
        if not batch_titles:
            return

        doc_texts = [f"query: {t} {a}" for t, a in zip(batch_titles, batch_abs)]
        vecs = enc.encode(doc_texts, batch_size=args.batch, normalize_embeddings=True)

        D, I = index.search(vecs.astype("float32"), args.topk)

        for meta, idx_row, score_row in zip(batch_meta, I, D):
            cand_ids   = [label_ids[i] for i in idx_row]
            cand_scores= 1 - score_row/2                     # back to cosine-sim

            if cross and args.rerank:
                # take first R candidates, build text pairs
                topR_ids  = cand_ids[:args.rerank]
                pairs     = [(meta["concat"], label_texts[cid]) for cid in topR_ids]
                rerank_scores = cross.predict(pairs, batch_size=16)
                # update first R scores, then resort
                cand_scores[:args.rerank] = rerank_scores
                top = np.argsort(cand_scores)[::-1][:args.topk]
                cand_ids = [cand_ids[i] for i in top]


            out_path.write(orjson.dumps({"id": meta["id"],
                                         "gnd_pred": cand_ids}).decode()+"\n")

        batch_titles.clear(); batch_abs.clear(); batch_meta.clear()

    for rec in tqdm.tqdm(ds_iter, unit="doc"):
        abstract = rec.get("abstract", "")
        if isinstance(abstract, list):          # handle array-style abstracts
            abstract = " ".join(abstract)
    
        title = rec.get("title", "")
        batch_titles.append(title)
        batch_abs.append(abstract)
        batch_meta.append({
            "id": rec["id"],
            "concat": f"{title} {abstract}"
        })
        if len(batch_titles) == args.batch:
            flush()



    flush()
    out_path.close()
    print("✅ Predictions written →", args.out)

# ──────────────────────────── CLI ──────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="dev.jsonl.zst or test.jsonl.zst")
    ap.add_argument("--index_dir", default="gnd_label_index")
    ap.add_argument("--out",  required=True, help="output JSONL for submission")
    ap.add_argument("--topk", type=int, default=50, help="dense k to return")
    ap.add_argument("--batch", type=int, default=64, help="doc encode batch")
    ap.add_argument("--rerank", type=int, default=0,
                    help="R>0 ⇒ rerank top-R with cross-encoder")
    args = ap.parse_args()
    main(args)
