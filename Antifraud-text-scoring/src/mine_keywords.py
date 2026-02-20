import json
import pandas as pd
from sklearn.model_selection import train_test_split
from features import clean_text


def search_fraud_keywords(df, top_n=50):

    df["clean"] = df["message"].apply(clean_text)

    fraud_texts = df.loc[df["label"] == 1, "clean"]
    words = " ".join(fraud_texts).split()

    freq = pd.Series(words).value_counts()

    return freq.head(top_n).index.tolist()


if __name__ == "__main__":

    df = pd.read_csv("data/sms.csv")

    X_train, _, y_train, _ = train_test_split(
        df["message"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    train_df = pd.DataFrame({
        "message": X_train,
        "label": y_train
    })

    keywords = search_fraud_keywords(train_df)

    with open("artifacts/fraud_keywords.json", "w") as f:
        json.dump(keywords, f)

    print("Fraud keywords saved.")
