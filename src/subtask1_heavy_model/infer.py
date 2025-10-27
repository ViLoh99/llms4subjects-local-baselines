#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""


Task-1 inference · logit ensemble + threshold tuning
"""

import argparse, json, orjson, zstandard as zstd, numpy as np, pathlib, torch, tqdm
from collections import defaultdict
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import f1_score
from src.common.tokenization import encode_title_abstract, get_tokenizer
from src.task1_cls.utils import collate


# ───────────────────────── helpers ──────────────────────────────────────────
def tensor_dict(batch, device):
    """Keep only tensor-valued keys and move to device."""
    return {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}


def load_code2idx(path):
    with open(path) as f:
        return json.load(f)                         # code ➜ 0…27


def stream_inputs(json_zst, code2idx, need_labels: bool):
    ds = load_dataset("json", data_files=json_zst, split="train", streaming=True)
    for rec in ds:
        enc = encode_title_abstract(rec["title"], rec["abstract"],
                                    rec["language"], rec["type"])
        enc["id"] = rec["id"]
        if need_labels:
            y = [0] * 28
            for c in rec["linsearch_codes"]:
                idx = code2idx.get(c)
                if idx is not None:
                    y[idx] = 1
            enc["labels"] = y
        yield enc


def build_dataloader(stream_gen, batch_size=16):
    batch = []
    for obj in stream_gen:
        batch.append(obj)
        if len(batch) == batch_size:
            yield collate(batch)
            batch = []
    if batch:
        yield collate(batch)


def load_model(ckpt_path):
    base = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/mdeberta-v3-base",
        num_labels=28,
        problem_type="multi_label_classification"
    ).half().cuda()                     # ← fp16 + GPU

    base.resize_token_embeddings(len(get_tokenizer()))
    model = PeftModel.from_pretrained(base, ckpt_path).half().cuda().eval()
    return model


def tune_thresholds(all_logits, all_labels):
    thresholds = []
    probs  = torch.sigmoid(torch.tensor(all_logits))
    labels = torch.tensor(all_labels)
    for j in range(28):
        best_t, best_f1 = 0.5, 0.0
        for t in np.linspace(0.05, 0.5, 10):
            f1 = f1_score(labels[:, j], (probs[:, j] >= t).int(), zero_division=0)
            if f1 > best_f1:
                best_t, best_f1 = t, f1
        thresholds.append(best_t)
    return thresholds


def per_label_f1(y_true, y_pred, code2idx):
    out = {}
    inv = {v: k for k, v in code2idx.items()}
    for j in range(28):
        f1 = f1_score(y_true[:, j], y_pred[:, j], zero_division=0)
        out[inv[j]] = round(float(f1), 5)
    return out


def save_eval(path, macro, micro, thresholds, per_label, preds):
    bundle = {
        "macro_f1": round(float(macro), 5),
        "micro_f1": round(float(micro), 5),
        "avg_threshold": round(float(np.mean(thresholds)), 5),
        "per_label_f1": per_label,
        #"predictions": preds              # comment if file size is an issue
    }
    pathlib.Path(path).write_text(json.dumps(bundle, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    print("💾 Eval + predictions saved →", path)


# ─────────────────────────── main ───────────────────────────────────────────
def main(args):
    code2idx  = load_code2idx(args.labels)
    need_gold = args.gold is not None

    print("🔄 Loading checkpoints …")
    models = [load_model(p) for p in args.ckpt]

    # ── calibration ──
    print("🔄 Calibrating thresholds on", args.calib)
    logits_list, labels_list = [], []
    calib_iter = stream_inputs(args.calib, code2idx, need_labels=True)

    for bi, batch in enumerate(tqdm.tqdm(build_dataloader(calib_iter, args.batch), unit="batch")):
        if args.max_batches and bi >= args.max_batches:
            break
        with torch.no_grad():
            model_inputs = tensor_dict(batch, models[0].device)
            logits = sum(m(**model_inputs).logits for m in models) / len(models)
        logits_list.append(logits.cpu())
        labels_list.append(batch["labels"].cpu())

    thresholds = tune_thresholds(torch.cat(logits_list),
                                 torch.cat(labels_list))
    print("✅ Thresholds ready (avg =", round(float(np.mean(thresholds)), 3), ")")

    # ── prediction ──
    preds_jsonl, gold_labels, preds_store = [], [], []
    target_iter = stream_inputs(args.input, code2idx, need_gold)

    for bi, batch in enumerate(tqdm.tqdm(build_dataloader(target_iter, args.batch), unit="batch")):
        if args.max_batches and bi >= args.max_batches:
            break
        with torch.no_grad():
            model_inputs = tensor_dict(batch, models[0].device)
            logits = sum(m(**model_inputs).logits for m in models) / len(models)

        probs = torch.sigmoid(logits).cpu().numpy()
        pred_bin = (probs >= thresholds).astype(int)
        
        # ---------- top-k fallback ----------
        if args.topk and args.topk > 0:
            # indices of the k largest probabilities per row
            topk_idx = np.argpartition(-probs, args.topk - 1, axis=1)[:, : args.topk]
            for row_bin, idxs in zip(pred_bin, topk_idx):
                row_bin[idxs] = 1
        # ------------------------------------


        for rid, vec, prb in zip(batch["id"], pred_bin, probs):
            rec = {"id": rid, "labels": vec.tolist(), "probs": prb.round(6).tolist()}
            preds_jsonl.append(orjson.dumps(rec).decode())
            preds_store.append(rec)

        if need_gold:
            gold_labels.extend(batch["labels"].cpu().numpy())

    # write predictions (plain JSONL)
    pathlib.Path(args.out).write_text("\n".join(preds_jsonl), encoding="utf-8")
    print("📝 Predictions written →", args.out)

    # ── metrics & optional save ──
    if need_gold and gold_labels:
        gold_arr = np.array(gold_labels)
        pred_arr = np.vstack([r["labels"] for r in preds_store])

        macro = f1_score(gold_arr, pred_arr, average="macro", zero_division=0)
        micro = f1_score(gold_arr, pred_arr, average="micro", zero_division=0)
        print(f"📊 Dev scores  macro={macro:.3f}  micro={micro:.3f}")

        if args.eval_out:
            per_lbl = per_label_f1(gold_arr, pred_arr, code2idx)
            save_eval(args.eval_out, macro, micro, thresholds, per_lbl, preds_store)


# ───────────────────────── CLI ──────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input",  required=True, help="JSONL.zst to predict on")
    p.add_argument("--calib",  required=True, help="Calibration split (dev)")
    p.add_argument("--labels", required=True, help="task1_code2idx.json")
    p.add_argument("--ckpt",   required=True, nargs="+", help="LoRA checkpoint dirs")
    p.add_argument("--out",    required=True, help="Predictions JSONL")
    p.add_argument("--gold",   help="dev.jsonl.zst (optional) for scoring")
    p.add_argument("--batch",  type=int, default=16)
    p.add_argument("--max_batches", type=int, help="Debug: stop after N batches")
    p.add_argument("--eval_out", help="Save metrics & predictions to JSON")
    p.add_argument("--topk", type=int, default=0, help="Always include the k highest-score labels (0 = disabled)")
    args = p.parse_args()
    main(args)
