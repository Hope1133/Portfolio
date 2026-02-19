import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from features import feauture_engineering


def load_data(path):
    df = pd.read_csv(path)
    df = df.iloc[:, :2].rename(columns={"v1": "label", "v2": "message"})
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df.dropna()

def NLP_vectorization(df: pd.DataFrame) -> pd.DataFrame:
    clean = df['message'].apply(clean_text)
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1,2), stop_words='english', min_df=2)
    X_tfidf = tfidf.fit_transform(clean)

    # Cнижаем размерность до 20 компонент
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_tfidf_reduced = svd.fit_transform(X_tfidf)

    # Создаём DataFrame с компонентами
    tfidf_cols = [f'tfidf_svd_{i}' for i in range(X_tfidf_reduced.shape[1])]
    df_tfidf = pd.DataFrame(X_tfidf_reduced, columns=tfidf_cols)

    return df_tfidf

def build_pipeline():
    text_transformer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2
    )

    numeric_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("text", text_transformer, "message"),
            ("num", numeric_transformer, ["text_len", "digit_count", "url_count", "phone_count"])
        ]
    )

    model = LogisticRegression(class_weight="balanced", max_iter=1000)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    return pipeline


def main():
    df = load_data("data/data.csv")
    df['label'] = df['label'].replace({'ham' : 0, 'spam' : 1})

    X = feauture_engineering(df.drop("label", axis=1))
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, "models/pipeline.pkl")


if __name__ == "__main__":
    main()




import re
import pandas as pd
from collections import Counter

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


def extract_numeric_features(X: pd.DataFrame) -> pd.DataFrame:
 Xc = X.copy()

 Xc['text_len'] = X['message'].str.len()
 Xc['num_words'] = X['message'].str.split().str.len()
 Xc['dig_num'] = X['message'].str.count(r"\d")
 Xc['caps_ratio'] = X['message'].apply(lambda x : sum([i.isupper() for i in x]) / len(x))
 Xc['num_special'] = X['message'].str.count(r"[^\w\s]")
 Xc['money_count'] = X['message'].str.count(r'\$|€|£|usd|eur').astype(int)
 Xc['url_count'] = X['message'].apply(lambda x : len(re.findall(url_pattern, x)))
 Xc['phone_count'] = X['message'].apply(lambda x : len(re.findall(phone_pattern, x)))

 return Xc[['text_len', 'num_words', 'dig_num', 'caps_ratio', 'num_special', 'money_count', 'url_count', 'phone_count']]


def count_spam_words(text: str, fraud_keywords: list) -> int:
    c = 0
    for word in text.split():
        if word in fraud_keywords:
            c+=1
    return(c)

def search_fraud_keywords(df: pd.DataFrame) -> list: 
    df['message'] = df['message'].apply(clean_text)
    spam_messanges = df.loc[df['label'].eq(1), 'message'].apply(lambda x : re.findall('[a-zA-Z]+', x))
    spam_words_count = Counter()
    for m in spam_messanges:
        spam_words_count.update(m)
    spam_words = set([el for el, count in spam_words_count.most_common(50)])

    non_spam_messanges = df.loc[df['label'].eq(0), 'message'].apply(lambda x : re.findall('[a-zA-Z]+', x))
    non_spam_words_count = Counter()
    for m in non_spam_messanges:
        non_spam_words_count.update(m)
    non_spam_words = set([el for el, count in non_spam_words_count.most_common(100)])

    fraud_keywords = spam_words - non_spam_words
    return fraud_keywords

def spam_words_feature(X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    X['spam_words_count'] = X['message'].apply(count_spam_words(search_fraud_keywords(df)))
    return X['spam_words_count']

def feature_engineering(X: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    X_for_train = extract_numeric_features(X)
    fraud_keywords = search_fraud_keywords(df)
    X_for_train['spam_words_count'] = X['message'].apply(count_spam_words(fraud_keywords))
    X_for_train = X_for_train.dropna().reset_index(drop=True) 

    return X_for_train
