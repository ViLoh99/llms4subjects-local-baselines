# src/common/preprocess_nltk.py
import re, unicodedata
from unidecode import unidecode
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

STOP_EN = set(stopwords.words("english"))
STOP_DE = set(stopwords.words("german"))
STEM_EN = SnowballStemmer("english")
STEM_DE = SnowballStemmer("german")
LEMMA_EN = WordNetLemmatizer()

def _clean(text: str) -> str:
    # lower-case, strip accents, keep letters/numbers
    text = unidecode(text.lower())
    text = re.sub(r"[^a-z0-9äöüß\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def preprocess(text: str, lang: str) -> str:
    text = _clean(text)
    tokens = nltk.word_tokenize(text, language="german" if lang.startswith("de") else "english")

    if lang.startswith("de"):
        tokens = [STEM_DE.stem(t) for t in tokens if t not in STOP_DE]
    else:  # english
        tokens = [LEMMA_EN.lemmatize(STEM_EN.stem(t)) for t in tokens if t not in STOP_EN]

    return " ".join(tokens)
