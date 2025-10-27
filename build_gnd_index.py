#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_gnd_index.py
Create embeddings + FAISS index for GND subject labels (Task 2).

Usage
-----
# full run
python build_gnd_index.py  subjects.json  --outdir gnd_label_index

# quick test on a single ID / text first
python build_gnd_index.py  subjects.json  --outdir gnd_label_index  --test
"""
import argparse, json, pathlib, sys
import numpy as np, faiss, tqdm
from sentence_transformers import SentenceTransformer

PREFIX_PASSAGE = "passage: "


# ---------- CONFIG ---------------------------------------------------------
DIM      = 1024        # E5-large vector size
CHUNK    = 20_000      # encode this many labels per GPU batch group
MODEL_ID = "intfloat/multilingual-e5-large-instruct"

# ---------- HELPERS --------------------------------------------------------
def iter_subjects(json_path):
    """
    Yield tuples  (gnd_id, enriched_text_for_embedding)
    Also collect the SAME enriched text so we can save it for reranking.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        raw_id = item.get("Code")            # e.g. 'gnd:4003694-7'
        if not raw_id:
            continue
        gnd_id = raw_id.split(":")[-1]       # '4003694-7'
        name   = item.get("Name", "")
        alt    = " | ".join(item.get("Alternate Name", []))
        cls    = item.get("Classification Name", "")
        defin  = item.get("Definition", "")

        parts  = [name]
        if alt:   parts.append(alt)
        if cls:   parts.append(cls)
        if defin: parts.append(f"Definition: {defin}")

        joined = ". ".join(parts).strip()
        if joined:
            yield gnd_id, f"{PREFIX_PASSAGE}{joined}", joined   # 3rd elem raw text
# ---------------------------------------------------------------------------
def build_index(json_path, outdir, batch_size, test_mode=False):
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) prepare model
    model = SentenceTransformer(MODEL_ID).half().cuda()

    # 2) optional quick smoke-test on 1 label
    if test_mode:
        gnd_id, text = next(iter_subjects(json_path))
        vec = model.encode(text, normalize_embeddings=True)
        print(f"Test OK → id={gnd_id}, |vec|={vec.shape}, first5={vec[:5]}")
        sys.exit(0)

    # 3) count rows
    total = sum(1 for _ in iter_subjects(json_path))
    print(f"⏳ Encoding {total:,} labels with {MODEL_ID} …")

    # 4) mem-map fp32 array on disk
    mmap = np.memmap(outdir / "label_emb_fp32.dat",
                     dtype="float32", mode="w+", shape=(total, DIM))
    gnd_ids = []
    all_rows = []             # store (gnd_id, emb_text, raw_text)

    row = 0
    batch_ids, batch_texts = [], []
    for gnd_id, emb_text, raw_text in tqdm.tqdm(iter_subjects(json_path), total=total, unit="label"):
        batch_ids.append(gnd_id)
        batch_texts.append(emb_text)
        all_rows.append((gnd_id, emb_text, raw_text))
        if len(batch_texts) == CHUNK:
            vecs = model.encode(batch_texts,
                                 batch_size=batch_size,
                                 normalize_embeddings=True)
            mmap[row:row+len(vecs)] = vecs
            row += len(vecs)
            gnd_ids.extend(batch_ids)
            batch_ids, batch_texts = [], []

    # flush remainder
    if batch_texts:
        vecs = model.encode(batch_texts,
                             batch_size=batch_size,
                             normalize_embeddings=True)
        mmap[row:row+len(vecs)] = vecs
        gnd_ids.extend(batch_ids)

    mmap.flush()
    print("✔️  Embeddings written:", mmap.filename)

    # 5) save ID list
    json.dump(gnd_ids, open(outdir / "label_ids.json", "w"))
    print("✔️  label_ids.json written")
    
    # 5b)  save human-readable label text for reranker
    label_text_path = outdir / "label_texts.json"
    with open(label_text_path, "w", encoding="utf-8") as f_out:
        json.dump({gid: txt for gid, _, txt in all_rows}, f_out, ensure_ascii=False)
    print("✔️  label_texts.json written")


    # 6) build FAISS IVF-PQ index
    index = faiss.index_factory(DIM, "IVF2048_HNSW32,PQ64")
    index.train(mmap)
    index.add(mmap)
    faiss.write_index(index, str(outdir / "faiss_ivfpq.index"))
    print("✅  FAISS index written:", outdir / "faiss_ivfpq.index")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("json",    help="subjects.json path (full GND label dump)")
    ap.add_argument("--outdir", default="gnd_label_index",
                    help="output directory (will be created)")
    ap.add_argument("--batch", type=int, default=128,
                    help="sentence-transformers encode() batch size")
    ap.add_argument("--test",  action="store_true",
                    help="run a single-embedding smoke test and exit")
    args = ap.parse_args()
    build_index(args.json, args.outdir, args.batch, test_mode=args.test)
