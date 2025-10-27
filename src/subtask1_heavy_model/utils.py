# src/task1_cls/utils.py
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from src.common.tokenization import encode_title_abstract, get_tokenizer

import torch

def streaming_dataset(jsonl_zst, code2idx):
    ds_stream = load_dataset("json", data_files=jsonl_zst,
                             split="train", streaming=True)
    for rec in ds_stream:
        # encode text
        enc = encode_title_abstract(rec["title"], rec["abstract"],
                                    rec["language"], rec["type"])
        # build 28-hot label vector
        y = [0]*28
        for code in rec["linsearch_codes"]:
            if code in code2idx:
                y[code2idx[code]] = 1
        enc["labels"] = y
        yield enc

def collate(batch):
    tok = get_tokenizer()
    # Separate out id/labels for later use; keep only tokenizer fields for pad()
    ids = [x.pop("id") for x in batch if "id" in x]
    labels = [x.pop("labels") for x in batch if "labels" in x]
    out = tok.pad(batch, return_tensors="pt")
    # Reattach ids/labels (as-is, not tensors)
    out["id"] = ids
    if labels:
        out["labels"] = torch.tensor(labels, dtype=torch.float)
    return out


def stream_records(jsonl_zst, code2idx):
    ds = load_dataset("json", data_files=jsonl_zst, split="train")
    records = []
    for rec in ds:
        enc = encode_title_abstract(rec["title"], rec["abstract"],
                                    rec["language"], rec["type"])
        labels = [0.0] * 28
        for c in rec["linsearch_codes"]:
            idx = code2idx.get(c)
            if idx is not None:
                labels[idx] = 1.0
        enc["labels"] = labels
        records.append(enc)
    return Dataset.from_list(records)