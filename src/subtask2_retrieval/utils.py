# src/task2_retrieval/utils.py
import json, numpy as np, faiss, torch, os
from sentence_transformers import SentenceTransformer
from src.common.tokenization import get_tokenizer   # reuse lang/type tokens

MODEL_ID = "intfloat/multilingual-e5-large-instruct"

''' #vers1
def load_label_index(index_dir="gnd_label_index"):
    index   = faiss.read_index(f"{index_dir}/faiss_ivfpq.index")
    ids     = json.load(open(f"{index_dir}/label_ids.json"))
    return index, ids
'''

def load_label_index(index_dir):
    """
    Return  (faiss_index, label_id_list, label_text_dict)
    """
    index = faiss.read_index(os.path.join(index_dir, "faiss_ivfpq.index"))
    label_ids = json.load(open(os.path.join(index_dir, "label_ids.json"), encoding="utf-8"))
    label_texts = json.load(open(os.path.join(index_dir, "label_texts.json"), encoding="utf-8"))
    return index, label_ids, label_texts


def get_encoder(device="cuda"):
    model = SentenceTransformer(MODEL_ID, device=device)
    model.max_seq_length = 512
    return model

def encode_doc(model, title, abstract, lang, rec_type):
    # E5 prompt style: "query: ... </s> document: ..."
    lang_tok = "[EN]" if lang.startswith("en") else "[DE]"
    txt = f"query: {lang_tok} {title}. {abstract}"
    vec = model.encode(txt, normalize_embeddings=True)
    return vec.astype("float32")


