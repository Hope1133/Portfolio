import re
import pandas as pd

url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+'
phone_pattern = r"\+?\d[\d\-\s]{7,}\d"

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(url_pattern, " URL ", text)
    text = re.sub(phone_pattern, " PHONE ", text)
    text = re.sub(r"\d+", " NUM ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["text_len"] = df["message"].str.len()
    df["digit_count"] = df["message"].str.count(r"\d")
    df["url_count"] = df["message"].str.count(r"http")
    df["phone_count"] = df["message"].str.count(phone_pattern)

    return df[["text_len", "digit_count", "url_count", "phone_count"]]
