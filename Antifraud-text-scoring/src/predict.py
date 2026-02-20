import pandas as pd
import joblib
from features import clean_text, extract_numeric_features


def main(input_path, output_path):
    model = joblib.load("models/pipeline.pkl")

    df = pd.read_csv(input_path)
    df["message"] = df["message"].apply(clean_text)

    numeric_features = extract_numeric_features(df)
    df = pd.concat([df, numeric_features], axis=1)

    df["proba"] = model.predict_proba(df)[:, 1]
    df["fraud_flag"] = (df["proba"] > 0.5).astype(int)

    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main("new_sms.csv", "scored_sms.csv")




---

tfidf = joblib.load("models/tfidf.pkl")
model = joblib.load("models/logreg_model.pkl")
