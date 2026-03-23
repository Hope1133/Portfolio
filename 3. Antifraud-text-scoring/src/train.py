import os
import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from features import clean_text, extract_numeric_features
import logging

logging.basicConfig(level=logging.INFO)

df = pd.read_csv("data/data.csv")
logging.info("Dataset loaded")
X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF
X_train_clean = X_train.apply(clean_text)
X_test_clean = X_test.apply(clean_text)

tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1,2), min_df=2)
X_train_tfidf = tfidf.fit_transform(X_train_clean)
X_test_tfidf = tfidf.transform(X_test_clean)
logging.info("TF-IDF fitted")
# Снижение размерности
# svd = TruncatedSVD(n_components=50, random_state=42)
# X_train_tfidf_reduced = svd.fit_transform(X_train_tfidf)
# X_test_tfidf_reduced = svd.transform(X_test_tfidf)

# Numeric features
X_train_num = extract_numeric_features(pd.DataFrame({"message": X_train}))
X_test_num = extract_numeric_features(pd.DataFrame({"message": X_test}))

scaler = StandardScaler()
X_train_num_scaled = scaler.fit_transform(X_train_num)
X_test_num_scaled = scaler.transform(X_test_num)

# Единая разреженная матрица
X_train_final = hstack([X_train_tfidf, X_train_num_scaled])
X_test_final = hstack([X_test_tfidf, X_test_num_scaled])

# Train model
logreg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
logreg.fit(X_train_final, y_train)
logging.info("Model training completed")
# Evaluation
y_pred = logreg.predict(X_test_final)

print(f'Precision: {precision_score(y_test, y_pred):.4f}')
print(f'Recall:    {recall_score(y_test, y_pred):.4f}')
print(f'F1-score:  {f1_score(y_test, y_pred):.4f}')

print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

# Save artifacts
os.makedirs("models", exist_ok=True)

joblib.dump(logreg, "models/logreg_model.pkl")
joblib.dump(tfidf, "models/tfidf.pkl")
# joblib.dump(svd, "models/svd.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("Training complete. Models saved.")