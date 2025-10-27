# src/common/tokenization.py
from transformers import AutoTokenizer


SPECIAL_TOKENS = [
    "[EN]", "[DE]",
    "[ART]", "[BOOK]", "[CONF]", "[REP]", "[THE]"
]

MODEL_NAME = "microsoft/mdeberta-v3-base"

_tokenizer = None   # lazy singleton


def get_tokenizer():
    """Return a tokenizer with our special tokens added (singleton)."""
    global _tokenizer
    if _tokenizer is None:
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        # add only if not already there (avoid double grow)
        new_tokens = [t for t in SPECIAL_TOKENS if t not in tok.vocab]
        if new_tokens:
            tok.add_special_tokens({"additional_special_tokens": new_tokens})
        _tokenizer = tok
    return _tokenizer
'''
#vers 1
def encode_title_abstract(title: str, abstract: str,
                          lang: str, rec_type: str,
                          max_len: int = 512):
    

    """
    Build a clean natural-language prefix for the model instead of special tokens.
    This improves compatibility with all tokenizers.

    lang: 'en' or 'de'
    rec_type: Article|Book|Conference|Report|Thesis
    """

    lang_str = "English" if lang.lower().startswith("en") else "German"

    type_map = {
        "Article": "article",
        "Book": "book",
        "Conference": "conference paper",
        "Report": "report",
        "Thesis": "thesis"
    }
    type_str = type_map.get(rec_type, "article")

    # Natural-language prefix
    prefix = f"This is a {lang_str} {type_str}."

    # Full text input
    full_text = f"{prefix} {title} </s> {abstract}"

    tok = get_tokenizer()
    return tok(
        full_text,
        truncation=True,
        padding=False,         # pad in collate_fn
        max_length=max_len
    )

'''
from src.common.preprocess_nltk import preprocess  # new import

def encode_title_abstract(title, abstract, lang, rec_type, max_len=512):
    # 1. NLTK preprocess
    title    = preprocess(title, lang)
    abstract = preprocess(abstract, lang)

    # 2. Natural-language prefix (Option-1 style, no special tokens)
    lang_str = "English" if lang.lower().startswith("en") else "German"
    type_map = {
        "Article":"article","Book":"book",
        "Conference":"conference paper","Report":"report","Thesis":"thesis"
    }
    prefix = f"This is a {lang_str} {type_map.get(rec_type,'article')}."
    full_text = f"{prefix} {title} </s> {abstract}"

    tok = get_tokenizer()        # unchanged
    return tok(full_text, truncation=True, padding=False, max_length=max_len)








if __name__ == "__main__":
    from datasets import load_dataset
    import json

    # Load a few examples from your dev file
    ds = load_dataset("json", data_files="dev.jsonl.zst", split="train")
    for i in range(3):
        rec = ds[i]
        print(f"\n📄 Sample {i+1} ID: {rec['id']}")
        tokens = encode_title_abstract(
            rec["title"],
            rec["abstract"],
            rec.get("language", "en"),
            rec.get("type", "Article")
        )
        print("🧩 Token IDs:", tokens["input_ids"][:30])
        print("🧩 Tokens:", get_tokenizer().convert_ids_to_tokens(tokens["input_ids"][:30]))
        print("📏 Token length:", len(tokens["input_ids"]))

print(preprocess("Modellierung sektorenübergreifender Dienstleistungen!", "de"))
# -> modellier sektorübergreif dienstleist

print(preprocess("Nanostructured diffraction gratings as polarizing beam splitters.", "en"))
# -> nanostructur diffract grat beam split

tokens = encode_title_abstract(
    title="Modellierung sektorenübergreifender Systemdienstleistungen bei gekoppelten Strom- und Gassektoren",
    abstract="Die Arbeit analysiert mögliche Schnittstellen und Optimierungspotentiale zwischen Strom- und Gasnetzen.",
    lang="de",
    rec_type="Article"
)

'''
print("📄 Tokens:", tokens.tokens)
print("📏 Token length:", len(tokens.input_ids))

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

# Models to test
ENCODER_MODELS = [
    "xlm-roberta-base",
    "bert-base-multilingual-cased",
    "distilbert-base-multilingual-cased",
]

SBERT_MODELS = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-base"
]

text = "This is a test sentence for multilingual tokenization."
full_text = f"This is a German article {text} </s> {text}"

print("\n=== Transformer Tokenizers ===\n")
for model_name in ENCODER_MODELS:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    enc = tokenizer(full_text, return_tensors="pt", truncation=True)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    print(f"{model_name} → {len(tokens)} tokens")
    print("Tokens:", tokens[:20], "\n")

print("\n=== Sentence-BERT Tokenizers ===\n")
for model_name in SBERT_MODELS:
    model = SentenceTransformer(model_name)
    tokenizer = model.tokenizer
    enc = tokenizer(full_text, return_tensors="pt", truncation=True)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    print(f"{model_name} → {len(tokens)} tokens")
    print("Tokens:", tokens[:20], "\n")

'''