from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import mlflow.sklearn

app = FastAPI(title="Fraud Detection API")

model = mlflow.sklearn.load_model("../models/random_forest_model2")

REAL_ORDER = list(model.feature_names_in_)


class Transaction(BaseModel):
    step: int
    type: int
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fraud Detection API</title>

        <style>
            body{
                font-family:Arial,sans-serif;
                background:#0f172a;
                color:white;
                text-align:center;
                padding-top:100px;
            }

            .card{
                width:700px;
                margin:auto;
                background:#1e293b;
                padding:40px;
                border-radius:20px;
                box-shadow:0 0 25px rgba(0,0,0,.3);
            }

            h1{
                color:#38bdf8;
            }

            .btn{
                display:inline-block;
                margin:15px;
                padding:15px 25px;
                border-radius:10px;
                text-decoration:none;
                color:white;
                font-weight:bold;
                background:linear-gradient(
                    90deg,
                    #2563eb,
                    #06b6d4
                );
            }

            .endpoint{
                background:#334155;
                padding:15px;
                border-radius:10px;
                margin-top:20px;
            }
        </style>
    </head>

    <body>

        <div class="card">

            <h1>💳 Fraud Detection API</h1>

            <p>
                Machine Learning Fraud Detection Service
            </p>

            <a class="btn" href="/docs">
                Swagger Docs
            </a>

            <a class="btn" href="/redoc">
                ReDoc
            </a>

            <div class="endpoint">
                POST /predict
            </div>

        </div>

    </body>
    </html>
    """
@app.post("/predict")
def predict(data: Transaction):
    try:

        row = {
            "step": data.step,
            "type": data.type,
            "amount": data.amount,
            "oldbalanceOrg": data.oldbalanceOrg,
            "newbalanceOrig": data.newbalanceOrig,
            "oldbalanceDest": data.oldbalanceDest,
            "newbalanceDest": data.newbalanceDest,
            "amount_log": np.log1p(data.amount),
            "amount_scaled": data.amount / 1000000,
            "balance_diff_org": data.oldbalanceOrg - data.newbalanceOrig,
            "balance_diff_dest": data.newbalanceDest - data.oldbalanceDest
        }

        X = pd.DataFrame([row])

        X = X[REAL_ORDER]

        proba = model.predict_proba(X)[0][1]
        pred = int(proba >= 0.3)

        return {
            "status": "success",
            "prediction": pred,
            "fraud_probability": float(proba)
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }