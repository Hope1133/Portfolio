import re, os
import pandas as pd

URL_PATTERN = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+'
PHONE_PATTERN = r"\+?\d[\d\-\s]{7,}\d"
MONEY_PATTERN = r'\$|€|£|usd|eur'

KEYWORDS_PATH = "artifacts/fraud_keywords.json"
if os.path.exists(KEYWORDS_PATH):
    with open(KEYWORDS_PATH, "r") as f:
        FRAUD_KEYWORDS = set(json.load(f))
else:
    FRAUD_KEYWORDS = set()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(URL_PATTERN, " URL ", text)
    text = re.sub(PHONE_PATTERN, " PHONE ", text)
    text = re.sub(r"\d+", " NUM ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def extract_numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    Xc = X.copy()
    text = Xc["message"].fillna("")

    features = pd.DataFrame(index=Xc.index)

    features["text_len"] = text.str.len()
    features["num_words"] = text.str.split().str.len()
    features["digit_count"] = text.str.count(r"\d")
    features["caps_ratio"] = text.apply(lambda x: sum(c.isupper() for c in x) / len(x) if len(x) > 0 else 0)

    features["special_count"] = text.str.count(r"[^\w\s]")
    features["money_count"] = text.str.count(MONEY_PATTERN)
    features["url_count"] = text.str.count(URL_PATTERN)
    features["phone_count"] = text.str.count(PHONE_PATTERN)

    features["spam_words_count"] = text.apply(lambda x: sum(word in FRAUD_KEYWORDS for word in x.split())
)


    return features
