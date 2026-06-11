import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import optuna

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

os.makedirs("models", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)

df = pd.read_csv("data/processed/processed.csv")

if "isFlaggedFraud" in df.columns:
    df = df.drop(columns=["isFlaggedFraud"])

fraud_df = df[df["isFraud"] == 1]
safe_df = df[df["isFraud"] == 0].sample(n=200000, random_state=42)

df = pd.concat([fraud_df, safe_df]).sample(frac=1, random_state=42).reset_index(drop=True)

df["amount_log"] = np.log1p(df["amount"])
df["amount_scaled"] = df["amount"] / (df["amount"].max() + 1)
df["balance_diff_org"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["balance_diff_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

X = df.drop(columns=["isFraud"])
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 250),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 6),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"])
    }

    model = RandomForestClassifier(
        **params,
        random_state=42,
        class_weight={0: 1, 1: 5},
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.3).astype(int)

    score = f1_score(y_test, preds, zero_division=0)

    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)

best_params = study.best_params

print("\nBest Params:", best_params)
print("Best F1:", study.best_value)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Fraud_Detection_Experiment")

with mlflow.start_run():

    model = RandomForestClassifier(
        **best_params,
        random_state=42,
        class_weight={0: 1, 1: 5},
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.3).astype(int)

    final_f1 = f1_score(y_test, preds, zero_division=0)

    for key, value in best_params.items():
        mlflow.log_param(key, value)

    mlflow.log_param("threshold", 0.3)
    mlflow.log_metric("f1_score", final_f1)

    mlflow.sklearn.log_model(model, "model")

mlflow.sklearn.save_model(model, "models/random_forest_model2")

print("\nTraining completed successfully.")
print("Model saved in models/random_forest_model2")