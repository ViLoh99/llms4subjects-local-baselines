# ──────────────────────────────────────────────────────────────
#  src/common/tokenization.py
#  Clean version – no custom special tokens, uses NLTK preprocess
# ──────────────────────────────────────────────────────────────
from functools import lru_cache
from transformers import AutoTokenizer
from src.common.preprocess_nltk import preprocess  # your NLTK cleaner

# You can switch to "xlm-roberta-base" or any other model easily
MODEL_NAME = "microsoft/mdeberta-v3-base"
MAX_LEN    = 512


# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_tokenizer():
    """
    Singleton tokenizer – no extra special tokens added.
    """
    return AutoTokenizer.from_pretrained(MODEL_NAME)


# ---------------------------------------------------------------------------
def encode_title_abstract(title: str,
                          abstract: str,
                          lang: str,
                          rec_type: str,
                          max_len: int = MAX_LEN):
    """
    • Preprocess title + abstract with NLTK (lowercase, stop-word drop, stem/lemma)
    • Build a short natural-language prefix instead of custom tokens
    • Return standard HuggingFace tokenization dict
    """
    # 1) NLTK cleaning
    title    = preprocess(title, lang)
    abstract = preprocess(abstract, lang)

    # 2) Human-readable prefix (helps model, causes no token splits)
    lang_str = "English" if lang.lower().startswith("en") else "German"
    type_map = {
        "Article":     "article",
        "Book":        "book",
        "Conference":  "conference paper",
        "Report":      "report",
        "Thesis":      "thesis"
    }
    prefix = f"This is a {lang_str} {type_map.get(rec_type, 'article')}."
    full_text = f"{prefix} {title} </s> {abstract}"

    tok = get_tokenizer()
    return tok(
        full_text,
        truncation=True,
        padding=False,            # batching code will pad
        max_length=max_len
    )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Quick sanity-check: prints token IDs & first 30 tokens for 3 samples
    (expects a local dev.jsonl.zst). Remove or modify for your own testing.
    """
    from datasets import load_dataset

    print("🔎 Debug tokenization on 3 samples …")
    ds = load_dataset("json", data_files="dev.jsonl.zst", split="train")
    tok = get_tokenizer()

    for i in range(3):
        rec = ds[i]
        enc = encode_title_abstract(
            rec["title"], rec["abstract"],
            rec.get("language", "en"), rec.get("type", "Article")
        )
        ids = enc["input_ids"]
        print(f"\nID={rec['id']}  len={len(ids)}")
        print("Tokens:", tok.convert_ids_to_tokens(ids[:30]), "…")
