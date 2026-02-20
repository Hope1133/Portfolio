import sys
import joblib
import pandas as pd
from scipy.sparse import hstack

from features import clean_text, extract_numeric_features

model = joblib.load("models/logreg_model.pkl")
tfidf = joblib.load("models/tfidf.pkl")
scaler = joblib.load("models/scaler.pkl")
# svd = joblib.load("models/svd.pkl")


def predict_messages(messages):
    """
    messages: list[str]
    returns: DataFrame with predictions + probabilities
    """

    df = pd.DataFrame({"message": messages})

    df["clean"] = df["message"].apply(clean_text)

    # X_text = svd.transform(tfidf.transform(df["clean"]))
    X_text = tfidf.transform(df["clean"])

    X_num = extract_numeric_features(df)
    X_num_scaled = scaler.transform(X_num)
    X_final = hstack([X_text, X_num_scaled])

    preds = model.predict(X_final)
    probs = model.predict_proba(X_final)[:, 1]

    results = df.copy()
    results["prediction"] = preds
    results["fraud_probability"] = probs

    return results

# CLI mode
if __name__ == "__main__":

    if len(sys.argv) > 1:
        # single message from command line
        message = " ".join(sys.argv[1:])
        output = predict_messages([message])
        print(output)

    else:
        print("Usage:")
        print("python predict.py 'Your SMS message here'")



