import os
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from pathlib import Path

from features import prepare_features
from dicts import calc_city_mean_salary, columns_to_drop, city_area_dict

RANDOM_STATE = 42
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

def load_data(path: str):
    df = pd.read_csv(path)
    X = df.drop("mean_salary", axis=1)
    y = df["mean_salary"]
    return X, y

def embeddings_model():
    MODEL_DIR.mkdir(exist_ok=True)  # создаём папку, если её нет
    model = SentenceTransformer("all-MiniLM-L6-v2")
    joblib.dump(model, MODEL_DIR / "emb_model.pkl")

def _linear(X, y_log):
    model = LinearRegression()
    model.fit(X, y_log)
    return model

def _XGB(X, y_log):
    model = XGBRegressor(
        random_state=RANDOM_STATE,
        colsample_bytree=0.6,
        gamma=0,
        learning_rate=0.1,
        max_depth=5,
        min_child_weight=5,
        n_estimators=100,
        subsample=0.8,
        n_jobs=-1
    )
    model.fit(X, y_log)
    return model

def save_model(model, feature_columns, model_name: str):
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / f"{model_name}.pkl")
    joblib.dump(feature_columns, MODEL_DIR / f"{model_name}_columns.pkl")


def main():
    embeddings_model()

    X, y = load_data(DATA_DIR / "train.csv")

    columns_to_drop(X)
    calc_city_mean_salary(pd.concat([X, y], axis=1))
    city_area_dict(X)
    X_prep = prepare_features(X)

    # pd.set_option('display.max_columns', None)
    # print(X_prep.iloc[:,:20])
    # return

    y_log = np.log1p(y)
    lin_model = _linear(X_prep, y_log)
    save_model(lin_model, X_prep.columns.tolist(), "linear_model")
    xgb_model = _XGB(X_prep, y_log)
    save_model(xgb_model, X_prep.columns.tolist(), "xgb_model")

    print("training completed.")


if __name__ == "__main__":
    main() 