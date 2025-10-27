import argparse, os, json
import numpy as np
import yaml, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_curve, f1_score
from utils import set_seed, load_split, load_labels, make_mlb, save_json

def fit_thresholds(y_true, y_prob):
    # y_true, y_prob: shape (n_samples, n_classes)
    thresholds = []
    for j in range(y_true.shape[1]):
        p, r, t = precision_recall_curve(y_true[:, j], y_prob[:, j])
        # F1 bei jedem Threshold (außer None)
        f1s = (2*p*r)/(p+r+1e-9)
        if len(t) == 0:
            thr = 0.5
        else:
            # PR liefert |t| = |p|-1; map zurück
            thr = float(t[np.nanargmax(f1s[:-1])])
        thresholds.append(thr)
    return thresholds

def apply_thresholds(y_prob, thresholds):
    return (y_prob >= np.array(thresholds)[None, :]).astype(int)

def main(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    set_seed(cfg.get("random_seed", 42))

    # Labels & Binarizer
    classes = load_labels(cfg["label_list_path"])
    mlb = make_mlb(classes)

    # Daten (nur Y hier, X embeddings laden wir aus .npy)
    _, y_train_raw = load_split(cfg["splits"]["train"], cfg["text_fields"], cfg["label_key"])
    _, y_dev_raw   = load_split(cfg["splits"]["dev"],   cfg["text_fields"], cfg["label_key"])


    Y_train = mlb.transform(y_train_raw)
    Y_dev   = mlb.transform(y_dev_raw)

    # Embeddings
    emb_dir = cfg["paths"]["emb_dir"]
    X_train = np.load(os.path.join(emb_dir, "train.npy"))
    X_dev   = np.load(os.path.join(emb_dir, "dev.npy"))

    # Klassifikator
    base = LogisticRegression(
        max_iter=1000, n_jobs=-1, solver="saga", class_weight="balanced", C=10.0
    )
    clf = OneVsRestClassifier(base, n_jobs=-1)
    clf.fit(X_train, Y_train)

    # Dev-Probs + Thresholds
    Y_dev_prob = clf.predict_proba(X_dev)
    thresholds = fit_thresholds(Y_dev, Y_dev_prob)

    # Quick dev F1
    Y_dev_hat = apply_thresholds(Y_dev_prob, thresholds)
    f1_micro = f1_score(Y_dev, Y_dev_hat, average="micro", zero_division=0)
    f1_macro = f1_score(Y_dev, Y_dev_hat, average="macro", zero_division=0)
    print(f"[DEV] micro-F1={f1_micro:.4f}  macro-F1={f1_macro:.4f}")

    # Speichern
    os.makedirs(os.path.dirname(cfg["paths"]["model_out"]), exist_ok=True)
    joblib.dump({"model": clf, "classes": classes}, cfg["paths"]["model_out"])
    save_json({ "thresholds": thresholds,
                "dev_f1_micro": f1_micro,
                "dev_f1_macro": f1_macro},
              cfg["paths"]["thresholds_out"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    main(args.config)
