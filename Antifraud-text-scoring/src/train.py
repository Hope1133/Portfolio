import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from features import clean_text, extract_numeric_features


def load_data(path):
    df = pd.read_csv(path)
    df = df.iloc[:, :2].rename(columns={"v1": "label", "v2": "message"})
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    return df.dropna()


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
    df = load_data("data/raw.csv")

    df["message"] = df["message"].apply(clean_text)
    numeric_features = extract_numeric_features(df)
    df = pd.concat([df, numeric_features], axis=1)

    X = df.drop("label", axis=1)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, "models/pipeline.pkl")


if __name__ == "__main__":
    main()
