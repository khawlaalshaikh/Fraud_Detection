import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

os.makedirs("models", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)

df = pd.read_csv("data/processed/processed.csv")
df['amount_log'] = np.log1p(df['amount'])
X = df.drop(columns=['isFraud'])
y = df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

mlflow.set_tracking_uri("file:///C:/Users/LOQ/Desktop/fraud-detection-mlops/mlruns")
mlflow.set_experiment("Fraud_Detection_Experiment")

with mlflow.start_run() as run:
    print("MLflow Run ID:", run.info.run_id)
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    mlflow.log_param("n_estimators", 300)
    mlflow.log_param("max_depth", 15)
    mlflow.log_param("min_samples_leaf", 3)
    mlflow.log_param("class_weight", "balanced")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(model, "random_forest_model")

mlflow.sklearn.save_model(model, "models/random_forest_model")
X_test.to_csv("data/processed/X_test.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Training done! Model saved to MLflow and models/random_forest_model for DVC.")