import argparse, os, json
import numpy as np
import yaml, joblib
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
from utils import set_seed, load_split, load_labels, make_mlb, save_json


def apply_thresholds(y_prob, thresholds):
    return (y_prob >= np.array(thresholds)[None, :]).astype(int)

def apply_thresholds_ensure1(y_prob, thresholds):
    preds = apply_thresholds(y_prob, thresholds)
    # ensure at least one positive per row
    empty = preds.sum(axis=1) == 0
    if empty.any():
        top1 = np.argmax(y_prob[empty], axis=1)
        preds[empty, top1] = 1
    return preds

def apply_global_threshold(y_prob, thr: float):
    return (y_prob >= thr).astype(int)

def apply_topk(y_prob, k: int):
    preds = np.zeros_like(y_prob, dtype=int)
    topk_idx = np.argpartition(-y_prob, kth=min(k-1, y_prob.shape[1]-1), axis=1)[:, :k]
    rows = np.arange(y_prob.shape[0])[:, None]
    preds[rows, topk_idx] = 1
    return preds

def apply_topk_with_minp(y_prob, k: int, p_min: float):
    preds = np.zeros_like(y_prob, dtype=int)
    # get sorted indices to filter by prob
    topk_idx = np.argpartition(-y_prob, kth=min(k-1, y_prob.shape[1]-1), axis=1)[:, :k]
    rows = np.arange(y_prob.shape[0])[:, None]
    mask = y_prob[rows, topk_idx] >= p_min
    preds[rows.repeat(k, axis=1)[mask], topk_idx[mask]] = 1
    # if none selected, fall back to top1
    empty = preds.sum(axis=1) == 0
    if empty.any():
        top1 = np.argmax(y_prob[empty], axis=1)
        preds[empty, top1] = 1
    return preds

def compute_metrics(y_true, y_pred):
    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

def main(cfg_path: str, split_name: str, mode: str, k: int, thr: float, p_min: float):
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg.get("random_seed", 42))

    classes = load_labels(cfg["label_list_path"])
    mlb = make_mlb(classes)

    split_path = cfg["splits"][split_name]
    _, y_raw = load_split(split_path, cfg["text_fields"], cfg["label_key"])
    Y_true = mlb.transform(y_raw)
    X = np.load(os.path.join(cfg["paths"]["emb_dir"], f"{split_name}.npy"))

    artefact = joblib.load(cfg["paths"]["model_out"])
    clf = artefact["model"]
    Y_prob = clf.predict_proba(X)

    suffix = ""
    if mode == "threshold":
        with open(cfg["paths"]["thresholds_out"], "r", encoding="utf-8") as f:
            thresholds = json.load(f)["thresholds"]
        Y_pred = apply_thresholds(Y_prob, thresholds)
    elif mode == "threshold_ensure1":
        with open(cfg["paths"]["thresholds_out"], "r", encoding="utf-8") as f:
            thresholds = json.load(f)["thresholds"]
        Y_pred = apply_thresholds_ensure1(Y_prob, thresholds)
        suffix = "_thrEnsure1"
    elif mode == "global":
        Y_pred = apply_global_threshold(Y_prob, thr)
        suffix = f"_global{thr:.2f}"
    elif mode == "topk":
        Y_pred = apply_topk(Y_prob, k)
        suffix = f"_top{k}"
    elif mode == "topk_minp":
        Y_pred = apply_topk_with_minp(Y_prob, k, p_min)
        suffix = f"_top{k}_p{p_min:.2f}"
    else:
        raise ValueError("mode must be one of: threshold | threshold_ensure1 | global | topk | topk_minp")

    metrics = compute_metrics(Y_true, Y_pred)
    print(f"[{split_name.upper()} {mode.upper()}{suffix}] {metrics}")

    report = classification_report(Y_true, Y_pred, target_names=classes, zero_division=0, output_dict=True)

    root, ext = os.path.splitext(cfg["paths"]["metrics_out"])
    out_path = f"{root}{suffix}{ext}"
    save_json({"metrics": metrics, "per_class": report}, out_path)
    print(f"Saved detailed report to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--split", default="test")  # or test_gold
    ap.add_argument("--mode", default="threshold",
                    choices=["threshold","threshold_ensure1","global","topk","topk_minp"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--thr", type=float, default=0.7)
    ap.add_argument("--p_min", type=float, default=0.40)
    args = ap.parse_args()
    main(args.config, args.split, args.mode, args.k, args.thr, args.p_min)
