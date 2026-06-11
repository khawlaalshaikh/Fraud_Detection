import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import learning_curve

os.makedirs("artifacts", exist_ok=True)
os.makedirs("mlruns", exist_ok=True)

X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

if "isFlaggedFraud" in X_test.columns:
    X_test = X_test.drop(columns=["isFlaggedFraud"])

model = mlflow.sklearn.load_model("models/random_forest_model2")

y_prob = model.predict_proba(X_test)[:, 1]
threshold = 0.3
y_pred = (y_prob >= threshold).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc = roc_auc_score(y_test, y_prob)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
print("ROC-AUC:", roc)

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Fraud_Detection_Experiment2")

with mlflow.start_run(run_name="evaluation"):

    mlflow.log_param("threshold", threshold)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.title("Confusion Matrix")

    cm_path = "artifacts/confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    mlflow.log_artifact(cm_path)

    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X_test,
        y_test,
        cv=3,
        scoring="f1",
        n_jobs=-1,
        train_sizes=np.linspace(0.2, 1.0, 5)
    )

    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, label="Training score")
    plt.plot(train_sizes, test_mean, label="Validation score")
    plt.xlabel("Training Size")
    plt.ylabel("F1 Score")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid()

    lc_path = "artifacts/learning_curve.png"
    plt.savefig(lc_path, bbox_inches="tight")
    plt.close()

    mlflow.log_artifact(lc_path)

print("\nEvaluation completed successfully.")