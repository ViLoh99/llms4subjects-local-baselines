import argparse, os
import numpy as np
import yaml
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer
from utils import set_seed, load_split

def prefix_e5(xs):
    return [f"query: {t}" for t in xs]

def embed_split(model, texts, batch_size=256, normalize=True, device="cuda"):
    all_vecs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        with torch.no_grad():
            emb = model.encode(
                batch,
                batch_size=len(batch),
                convert_to_numpy=True,
                device=device,
                show_progress_bar=False,
                normalize_embeddings=normalize
            )
        all_vecs.append(emb)
    return np.concatenate(all_vecs, axis=0)

def main(cfg_path: str, only: str = None):
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg.get("random_seed", 42))
    os.makedirs(cfg["paths"]["emb_dir"], exist_ok=True)

    model_name = cfg["embedding"]["model_name"]
    normalize  = cfg["embedding"]["normalize"]
    e5_pref    = cfg["embedding"].get("e5_use_prefix", True)
    bs         = cfg["embedding"]["batch_size"]
    device     = "cuda" if torch.cuda.is_available() else "cpu"

    model = SentenceTransformer(model_name, device=device)

        # choose which splits to run
    splits = cfg["splits"]
    to_run = splits.items() if not only else [(only, splits[only])]
    for split_name, path in to_run:
        texts, _ = load_split(path, cfg["text_fields"], cfg["label_key"])
        if "intfloat/multilingual-e5" in model_name and e5_pref:
            texts = prefix_e5(texts)
        vecs = embed_split(model, texts, batch_size=bs, normalize=normalize, device=device)
        np.save(os.path.join(cfg["paths"]["emb_dir"], f"{split_name}.npy"), vecs)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--only", choices=["train","dev","test","test_gold"])
    args = ap.parse_args()
    main(args.config, args.only)
