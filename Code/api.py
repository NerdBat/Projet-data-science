from pathlib import Path
import json

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal


# =========================================================
# Configuration
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MLFLOW_DB = (BASE_DIR / "mlflow.db").as_posix()

MODEL_URI = "models:/Churn_Classifier@production"
LOCAL_MODEL_PATH = ARTIFACTS_DIR / "best_model.joblib"
CONFIG_PATH = ARTIFACTS_DIR / "model_config.json"

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")

app = FastAPI(
    title="Churn Prediction API",
    description="API de prédiction du churn client",
    version="1.0.0"
)

model = None
model_source = None
model_config = None


# =========================================================
# Schéma des données
# =========================================================
class CustomerData(BaseModel):
    gender: Literal["Male", "Female"]
    age: int = Field(..., ge=18, le=100)
    tenure_months: int = Field(..., ge=0)
    contract_type: Literal["Monthly", "Quarterly", "Yearly"]
    monthly_logins: int = Field(..., ge=0)
    weekly_active_days: int = Field(..., ge=0, le=7)
    avg_session_time: float = Field(..., ge=0)
    monthly_fee: float = Field(..., ge=0)
    payment_failures: int = Field(..., ge=0)
    nps_score: int = Field(..., ge=-100, le=100)


# =========================================================
# Chargement modèle + config
# =========================================================
def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Fichier config introuvable : {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model():
    global model, model_source, model_config

    model_config = load_config()

    # 1) on essaie MLflow
    try:
        model = mlflow.sklearn.load_model(MODEL_URI)
        model_source = "mlflow"
        print("Modèle chargé depuis MLflow.")
        return
    except Exception as e:
        print(f"Chargement MLflow impossible : {e}")

    # 2) fallback local
    if LOCAL_MODEL_PATH.exists():
        model = joblib.load(LOCAL_MODEL_PATH)
        model_source = "joblib"
        print("Modèle chargé depuis le fichier local best_model.joblib.")
        return

    raise RuntimeError("Impossible de charger le modèle depuis MLflow ou joblib.")


@app.on_event("startup")
def startup_event():
    load_model()


# =========================================================
# Routes
# =========================================================
@app.get("/")
def home():
    return {
        "message": "API de prédiction du churn client",
        "endpoints": ["/health", "/model-info", "/predict"]
    }


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")

    return {
        "status": "ok",
        "model_loaded": True,
        "model_source": model_source
    }


@app.get("/model-info")
def get_model_info():
    if model_config is None:
        raise HTTPException(status_code=500, detail="Configuration du modèle non chargée")

    return {
        "model_name": model_config.get("model_name"),
        "threshold": model_config.get("threshold"),
        "features": model_config.get("features"),
        "target": model_config.get("target"),
        "model_source": model_source
    }


@app.post("/predict")
def predict(data: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")

    try:
        input_dict = data.model_dump()
    except AttributeError:
        input_dict = data.dict()

    try:
        features = model_config["features"]
        threshold = float(model_config.get("threshold", 0.5))

        df = pd.DataFrame([input_dict])

        # on remet les colonnes dans le bon ordre
        df = df[features]

        proba = float(model.predict_proba(df)[0][1])
        prediction = int(proba >= threshold)

        if prediction == 1:
            risk_level = "élevé" if proba >= 0.70 else "modéré"
        else:
            risk_level = "faible"

        return {
            "prediction": prediction,
            "churn_probability": round(proba, 4),
            "threshold_used": threshold,
            "risk_level": risk_level,
            "model_name": model_config.get("model_name"),
            "model_source": model_source
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur lors de la prédiction : {str(e)}")