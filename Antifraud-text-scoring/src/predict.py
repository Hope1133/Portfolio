import sys
import joblib
import pandas as pd
from scipy.sparse import hstack

from features import clean_text, extract_numeric_features


# ======================
# 1. Load artifacts
# ======================

MODEL_PATH = "models/logreg_model.pkl"
TFIDF_PATH = "models/tfidf.pkl"

model = joblib.load(MODEL_PATH)
tfidf = joblib.load(TFIDF_PATH)


# ======================
# 2. Prediction function
# ======================

def predict_messages(messages):
    """
    messages: list[str]
    returns: DataFrame with predictions + probabilities
    """

    df = pd.DataFrame({"message": messages})

    # --- text preprocessing
    df["clean"] = df["message"].apply(clean_text)

    X_text = tfidf.transform(df["clean"])

    # --- numeric features
    X_num = extract_numeric_features(df)
    X_num = X_num.values

    # --- combine
    X_final = hstack([X_text, X_num])

    # --- predict
    preds = model.predict(X_final)
    probs = model.predict_proba(X_final)[:, 1]

    results = df.copy()
    results["prediction"] = preds
    results["fraud_probability"] = probs

    return results


# ======================
# 3. CLI mode
# ======================

if __name__ == "__main__":

    if len(sys.argv) > 1:
        # single message from command line
        message = " ".join(sys.argv[1:])
        output = predict_messages([message])
        print(output)

    else:
        print("Usage:")
        print("python predict.py 'Your SMS message here'")

# import pandas as pd
# import joblib
# from features import clean_text, extract_numeric_features


# def main(input_path, output_path):
#     model = joblib.load("models/pipeline.pkl")

#     df = pd.read_csv(input_path)
#     df["message"] = df["message"].apply(clean_text)

#     numeric_features = extract_numeric_features(df)
#     df = pd.concat([df, numeric_features], axis=1)

#     df["proba"] = model.predict_proba(df)[:, 1]
#     df["fraud_flag"] = (df["proba"] > 0.5).astype(int)

#     df.to_csv(output_path, index=False)


# if __name__ == "__main__":
#     main("new_sms.csv", "scored_sms.csv")





