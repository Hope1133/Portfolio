import json
import pandas as pd
from sklearn.model_selection import train_test_split
from features import clean_text
from collections import Counter
import re

def search_fraud_keywords(df: pd.DataFrame) -> list:
    df["clean"] = df["message"].apply(clean_text)

    spam_messanges = df.loc[df['label'].eq(1), 'clean'].apply(lambda x : re.findall('[a-zA-Z]+', x))
    spam_words_count = Counter()
    for m in spam_messanges:
        spam_words_count.update(m)
    spam_words = set([el for el, count in spam_words_count.most_common(50)])

    non_spam_messanges = df.loc[df['label'].eq(0), 'clean'].apply(lambda x : re.findall('[a-zA-Z]+', x))
    non_spam_words_count = Counter()
    for m in non_spam_messanges:
        non_spam_words_count.update(m)
    non_spam_words = set([el for el, count in non_spam_words_count.most_common(100)])

    fraud_keywords = spam_words - non_spam_words

    return list(fraud_keywords)


if __name__ == "__main__":

    df = pd.read_csv("data/data.csv")
    X_train, X_test, y_train, y_test = train_test_split(df.drop('label', axis=1), df['label'], test_size=0.2, random_state=42, stratify=df['label'])

    train_df = X_train.copy()
    train_df["label"] = y_train.values

    fraud_keywords = search_fraud_keywords(train_df)

    with open("artifacts/fraud_keywords.json", "w") as f:
        json.dump(fraud_keywords, f)

    print("Fraud keywords saved.")


