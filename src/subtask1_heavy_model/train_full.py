#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""


python src/task1_cls/train_full.py \
       --train train.jsonl.zst \
       --dev   dev.jsonl.zst \
       --labels task1_code2idx.json \
       --outdir checkpoints/task1_full_seed17 \
       --seed 17
"""

import argparse, json, os
import numpy as np, torch
from datasets import load_dataset, Dataset
from transformers import (AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, set_seed)
from sklearn.metrics import f1_score
from src.common.tokenization import encode_title_abstract, get_tokenizer

# ──────────────────────────────────────────────────────────────
#  CONFIG – tweak here only
MAX_RECORDS   = None          # small debug train set (set None for full)
UNFREEZE_LAYERS = 4            # last N transformer layers; 0 = head-only,
                               # 999 = full fine-tune if GPU allows
FULL_FINETUNE = False          # set True to ignore UNFREEZE_LAYERS and train all
BATCH_SIZE    = 4              # fits 8 GB VRAM in fp16
GRAD_ACCUM    = 4              # effective batch 16
EPOCHS        = 2
# ──────────────────────────────────────────────────────────────


def load_code2idx(path):
    with open(path) as f:
        return json.load(f)

def stream_records(jsonl_zst, code2idx):
    ds = load_dataset("json", data_files=jsonl_zst, split="train")
    if MAX_RECORDS and len(ds) > MAX_RECORDS:
        ds = ds.shuffle(seed=42).select(range(MAX_RECORDS))

    records = []
    for rec in ds:
        enc = encode_title_abstract(
            rec["title"], rec["abstract"], rec["language"], rec["type"])
        labels = [0.0]*28
        for c in rec["linsearch_codes"]:
            idx = code2idx.get(c)
            if idx is not None:
                labels[idx] = 1.0
        enc["labels"] = labels
        records.append(enc)
    return Dataset.from_list(records)

def collate(batch):
    tok = get_tokenizer()
    out = tok.pad(batch, return_tensors="pt")
    out["labels"] = out["labels"].float()
    return out

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) >= 0.5).int().numpy()
    macro = f1_score(labels, preds, average="macro", zero_division=0)
    micro = f1_score(labels, preds, average="micro", zero_division=0)
    return {"macro_F1": macro, "micro_F1": micro}

# ──────────────────────────────────────────────────────────────
def main(args):
    set_seed(args.seed)
    tok = get_tokenizer()

    code2idx = load_code2idx(args.labels)
    train_ds = stream_records(args.train, code2idx)
    dev_ds   = stream_records(args.dev,   code2idx)

    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/mdeberta-v3-base",
        num_labels=28,
        problem_type="multi_label_classification"
    )
    model.resize_token_embeddings(len(tok))   # ensure prefix tokens if any

    # ── freeze / unfreeze logic ───────────────────────────────
    for p in model.parameters():
        p.requires_grad_(False)

    # always train classifier
    for p in model.classifier.parameters():
        p.requires_grad_(True)

    if FULL_FINETUNE:
        for p in model.parameters():
            p.requires_grad_(True)
    elif UNFREEZE_LAYERS > 0:
        for layer in model.deberta.encoder.layer[-UNFREEZE_LAYERS:]:
            for p in layer.parameters():
                p.requires_grad_(True)

    # move to GPU & enable fp16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.gradient_checkpointing_enable()

    # count trainables
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"🧮 Trainable params: {trainable:,}/{total:,} "
          f"({trainable/total*100:.2f}%)")

    # ── training args ─────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.outdir,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=EPOCHS,
        learning_rate=3e-5,
        weight_decay=0.01,
        fp16=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_F1",
        seed=args.seed,
        report_to="none",
        label_names=["labels"]
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collate,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.outdir)
    tok.save_pretrained(args.outdir)

    print(trainer.evaluate())

# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train",  required=True)
    ap.add_argument("--dev",    required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed",   type=int, default=42)
    main(ap.parse_args())
