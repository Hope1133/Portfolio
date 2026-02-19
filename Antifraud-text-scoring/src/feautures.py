import re
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def clean_text(text: str) -> str:
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+'
    phone_pattern = r"\+?\d[\d\-\s]{7,}\d"

    text = text.lower()
    text = re.sub(url_pattern, " URL ", text)
    text = re.sub(phone_pattern, " PHONE ", text)
    text = re.sub(r"\d+", " NUM ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df['text_len'] = df['message'].str.len()
    df['word_num'] = df['message'].str.split().apply(lambda x: len(x))
    df['dig_num'] = df['message'].str.count(r"\d")
    df['caps_ratio'] = df['message'].apply(lambda x : sum([i.isupper() for i in x]) / len(x))
    df['num_special'] = df['message'].str.count(r"[^\w\s]")
    df['num_words'] = df['message'].str.split().str.len()
    df['num_money'] = df['message'].str.count(r'\$|€|£|usd|eur').astype(int)

    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+])+'
    df['url_count'] = df['message'].apply(lambda x : len(re.findall(url_pattern, x)))
    # df['urls'] = df['message'].apply(lambda x : re.findall(url_pattern, x))

    phone_pattern = r"\+?\d[\d\-\s]{7,}\d"
    df['phone_count'] = df['message'].apply(lambda x : len(re.findall(phone_pattern, x)))

    return df[['text_len', 'word_num', 'dig_num', 'caps_ratio', 'num_special', 'num_words', 'num_money', 'url_count', 'phone_count']]


def count_spam_words(text: str, fraud_keywords: list) -> int:
    c = 0
    for word in text.split():
        if word in fraud_keywords:
            c+=1
    return(c)

def spam_words_feature(df: pd.DataFrame) -> pd.DataFrame:
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

    df['spam_words_count'] = df['message'].apply(count_spam_words(fraud_keywords))

    return df['spam_words_count']

def NLP_vectorization(df: pd.DataFrame) -> pd.DataFrame:
    df['message'] = df['message'].apply(clean_text)
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1,2), stop_words='english', min_df=2)
    X_tfidf = tfidf.fit_transform(df['message'])

    # Cнижаем размерность до 20 компонент
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_tfidf_reduced = svd.fit_transform(X_tfidf)

    # Создаём DataFrame с компонентами
    tfidf_cols = [f'tfidf_svd_{i}' for i in range(X_tfidf_reduced.shape[1])]
    df_tfidf = pd.DataFrame(X_tfidf_reduced, columns=tfidf_cols)

    return df_tfidf

def feauture_engineering(df: pd.DataFrame) -> pd.DataFrame:
    numeric_features = extract_numeric_features(df)
    fraud_features = spam_words_feature(df)
    nlp_features = NLP_vectorization(df)
    df_for_train = pd.concat([numeric_features, fraud_features, nlp_features], axis=1)
    df_for_train = df_for_train.dropna().reset_index(drop=True) 