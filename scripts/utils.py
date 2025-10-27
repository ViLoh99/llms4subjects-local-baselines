import json, random, os
import numpy as np
import torch
from typing import List, Dict
from sklearn.preprocessing import MultiLabelBinarizer

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def load_split(path: str, text_fields: List[str], label_key: str):
    X, Y = [], []
    for rec in read_jsonl(path):
        parts = [str(rec.get(f, "")).strip() for f in text_fields]
        text = (" [SEP] ".join([p for p in parts if p])).strip()
        X.append(text)
        # label_key = "linsearch_codes" or "gnd_subject_ids"
        Y.append([str(x) for x in rec.get(label_key, [])])
    return X, Y


def load_labels(label_path: str) -> List[str]:
    with open(label_path, "r", encoding="utf-8") as f:
        labels = [ln.strip() for ln in f if ln.strip()]
    return labels

def make_mlb(classes: List[str]) -> MultiLabelBinarizer:
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([classes])  # erzwingt feste Reihenfolge
    return mlb

def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
