#!/usr/bin/env python
"""
merge_lora.py — merge a LoRA checkpoint into the base DeBERTa model
and save it as a standalone HuggingFace model.


Example:
python src/task1_cls/merge_lora.py \
    --ckpt checkpoints/task1_seed17/checkpoint-8457 \
    --out  merged_models/task1_seed17
"""
import argparse, pathlib, torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.common.tokenization import get_tokenizer


def main(a):
    print("🔄 Loading base …")
    base = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/mdeberta-v3-base",
        num_labels=28,
        problem_type="multi_label_classification"
    )

    # resize to match training vocab
    tok = get_tokenizer()
    base.resize_token_embeddings(len(tok))

    print("🔄 Loading LoRA …")
    model = PeftModel.from_pretrained(base, a.ckpt)

    print("🔗 Merging …")
    model = model.merge_and_unload()    # now a plain HF model
    model.eval()

    print("💾 Saving merged model →", a.out)
    pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(a.out)
    AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base").save_pretrained(a.out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="LoRA checkpoint dir")
    ap.add_argument("--out",  required=True, help="output dir for merged model")
    args = ap.parse_args()
    main(args)
