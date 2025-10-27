from transformers import AutoTokenizer

'''

samples = [
    {
        "id": "3A175995943X",
        "title": "Modellierung sektorenübergreifender Systemdienstleistungen bei gekoppelten Strom- und Gassektoren",
        "abstract": "Die Arbeit analysiert gekoppelte Energiesysteme, in denen Strom- und Gasnetzbetreiber durch sektorübergreifende Dienstleistungen gemeinsam zur Netzstabilität beitragen.",
        "lang": "de",
        "type": "Article"
    },
    {
        "id": "3A1698439806",
        "title": "Nano-structured diffraction gratings as polarizing beam splitters under vertical incidence",
        "abstract": "Polarizing beam splitters have numerous applications in optical systems. We present a design and manufacturing process for a nanostructured diffraction grating with optimized diffraction efficiencies.",
        "lang": "en",
        "type": "Article"
    }
]

def build_input_string(title, abstract, lang, rec_type):
    lang_str = "English" if lang.lower().startswith("en") else "German"
    type_map = {
        "Article": "article",
        "Book": "book",
        "Conference": "conference paper",
        "Report": "report",
        "Thesis": "thesis"
    }
    type_str = type_map.get(rec_type, "article")
    return f"This is a {lang_str} {type_str}. {title} </s> {abstract}"

tokenizer_models = {
    "xlm-roberta-base": AutoTokenizer.from_pretrained("xlm-roberta-base"),
    "bert-base-multilingual-cased": AutoTokenizer.from_pretrained("bert-base-multilingual-cased"),
    "distilbert-base-multilingual-cased": AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased"),
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": AutoTokenizer.from_pretrained("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
    "intfloat/multilingual-e5-base": AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base"),
}

for model_name, tokenizer in tokenizer_models.items():
    print(f"\n=== {model_name} ===")
    for sample in samples:
        text = build_input_string(sample["title"], sample["abstract"], sample["lang"], sample["type"])
        tokens = tokenizer.tokenize(text)
        print(f"Sample ID: {sample['id']}")
        print(f"Token count: {len(tokens)}")
        print(f"Tokens: {tokens[:30]}")  # show first 30 tokens
        
from src.common.tokenization import encode_title_abstract

tokens = encode_title_abstract(
    title="Modellierung sektorenübergreifender Systemdienstleistungen bei gekoppelten Strom- und Gassektoren",
    abstract="Die Arbeit analysiert mögliche Schnittstellen und Optimierungspotentiale zwischen Strom- und Gasnetzen.",
    lang="de",
    rec_type="Article"
)

print("📄 Tokens:", tokens.tokens)
print("📏 Token length:", len(tokens.input_ids))
        '''
        
        
import nltk
from transformers import AutoTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re

nltk.download("stopwords")

def preprocess(text, lang):
    text = text.lower()
    text = re.sub(r"[^a-zA-ZäöüÄÖÜß\s]", "", text)
    tokens = text.split()
    stop_words = set(stopwords.words("german" if lang == "de" else "english"))
    stemmer = SnowballStemmer("german" if lang == "de" else "english")
    return " ".join(stemmer.stem(t) for t in tokens if t not in stop_words)

samples = {
    "3A175995943X": {
        "title": "Modellierung sektorenübergreifender Systemdienstleistungen bei gekoppelten Strom- und Gassektoren",
        "abstract": "Die Arbeit analysiert verschiedene Ansätze zur Modellierung von Systemdienstleistungen unter besonderer Berücksichtigung der Sektorkopplung.",
        "lang": "de"
    },
    "3A1698439806": {
        "title": "Nano-structured diffraction gratings as polarizing beam splitters under vertical incidence",
        "abstract": "Polarizing beam splitters have numerous applications in optical systems. We present a design and manufacturing process for a nanostructured diffraction grating with optimized diffraction efficiencies.",
        "lang": "en"
    }
}

model_names = [
    "xlm-roberta-base",
    "bert-base-multilingual-cased",
    "distilbert-base-multilingual-cased",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "intfloat/multilingual-e5-base"
]

from transformers import AutoTokenizer

for model in model_names:
    tokenizer = AutoTokenizer.from_pretrained(model)
    print(f"\n=== {model} ===")
    for sid, sample in samples.items():
        #pre_title = preprocess(sample["title"], sample["lang"])
        #pre_abs = preprocess(sample["abstract"], sample["lang"])
        #print ('TITEL!!!:::    ' + pre_title)
        #print ('ABSTRACT!!!!::::   ' + pre_abs)
        pre_title = (sample["title"], sample["lang"])
        pre_abs = (sample["abstract"], sample["lang"])
        text = f"This is a {'German' if sample['lang'] == 'de' else 'English'} article. {pre_title} </s> {pre_abs}"
        tokens = tokenizer.tokenize(text)
        print(f"Sample ID: {sid}")
        print(f"Token count: {len(tokens)}")
        print(f"Tokens: {tokens[:30]}{' ...' if len(tokens) > 30 else ''}")
        
