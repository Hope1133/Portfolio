from features import prepare_features
import pandas as pd
import numpy as np
import joblib

def pred_cols_coincedence(X_train, X_pred):
    '''
    Проверка, что колонки совпадают с тренировочным набором
    Добавление отсутствующих колонок
    ---
    Returns
    X_pred - обновленный с правильными колонками
    '''

    train_cols = set(X_train.columns)
    test_cols = set(X_pred.columns)

    for col in train_cols - test_cols:
        X_pred[col] = 0
        print(col)

    for col in test_cols - train_cols:
        X_pred.drop(col, axis=1, inplace=True)

    # Упорядочивание колонок как в тренировочном наборе
    train_cols_order = X_train.columns.tolist()
    return X_pred[train_cols_order]

df = pd.read_csv("data/train_contest.csv")
df_pred = pd.read_csv("data/for_prediction.csv")

X = df.drop('mean_salary', axis=1)
y_log = np.log1p(df['mean_salary'])

X_pred = df_pred.drop('Id', axis=1)

X = prepare_features(X)
X_pred = prepare_features(X_pred)

X_pred = pred_cols_coincedence(X, X_pred)

# base_reg = LinearRegression().fit(X, y_log)
base_reg = joblib.load("models/linear_model.pkl")

prediction_log = base_reg.predict(X_pred)
prediction = np.expm1(prediction_log) 
submission = pd.DataFrame({
    'Id': df_pred['Id'],
    'Predicted': prediction
})

submission.to_csv('submissions/submission_XGBR.csv', index=False)

# XGBR = XGBRegressor(random_state=42, colsample_bytree=0.6, gamma=0, learning_rate=0.1, max_depth=5, min_child_weight=5, n_estimators=100, subsample=0.8).fit(X, y_log)
XGBR = joblib.load("models/xgb_model.pkl")

prediction_log = XGBR.predict(X_pred)
prediction = np.expm1(prediction_log) 
submission = pd.DataFrame({
    'Id': df_pred['Id'],
    'Predicted': prediction
})

submission.to_csv('submissions/submission_XGBR.csv', index=False)