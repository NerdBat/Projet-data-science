from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import mlflow.sklearn
import os

app = FastAPI(title="Churn Prediction API")

# 1. Configuration de l'accès dynamique à MLflow [cite: 89, 435]
# On utilise l'alias 'Production' qui est géré par ton script de promotion automatique.
MODEL_NAME = "Churn_Classifier"
MODEL_ALIAS = "production" 
try:
    # On charge le modèle via l'URI utilisant l'alias @Production [cite: 136]
    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"✅ Modèle {MODEL_NAME} (Alias: {MODEL_ALIAS}) chargé avec succès.")
except Exception as e:
    print(f"❌ Erreur critique lors du chargement depuis MLflow Registry : {e}")
    # Indispensable pour la gestion des erreurs lors de l'industrialisation 
    model = None

# 2. Définition complète du format de données (EF5) [cite: 95, 211, 441]
# Note : Toutes les colonnes utilisées par le pipeline de preprocessing doivent être ici.
class CustomerData(BaseModel):
    gender: str
    age: int
    tenure_months: int
    contract_type: str
    monthly_logins: int
    weekly_active_days: int
    avg_session_time: float
    monthly_fee: float
    payment_failures: int
    nps_score: int
    # Ajoute ici les colonnes restantes si ton dataset en contient d'autres (ex: 'total_revenue')

@app.get("/health")
def health_check():
    """Vérifie que le service et le modèle sont opérationnels [cite: 96, 443]"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {
        "status": "healthy", 
        "model_source": "MLflow Registry",
        "model_name": MODEL_NAME,
        "alias": MODEL_ALIAS
    }

@app.post("/predict")
def predict(data: CustomerData):
    """Reçoit un JSON et renvoie la probabilité de churn [cite: 95, 441]"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non disponible")
        
    try:
        # Conversion du JSON en DataFrame pour le pipeline [cite: 101, 435]
        df_input = pd.DataFrame([data.dict()])
        
        # Inférence (le pipeline MLflow gère le scaling et l'encodage automatiquement) [cite: 130, 435]
        probability = model.predict_proba(df_input)[0][1]
        
        # Seuil de décision métier ajusté pour maximiser le Recall [cite: 308, 345]
        # Dans un contexte de churn, on préfère alerter à partir de 30% de probabilité.
        threshold = 0.3 
        prediction = 1 if probability >= threshold else 0
        
        return {
            "prediction": prediction,
            "churn_probability": round(float(probability), 4),
            "applied_threshold": threshold,
            "status": "success"
        }
    except Exception as e:
        # Gestion rigoureuse des erreurs pour l'industrialisation 
        raise HTTPException(status_code=400, detail=str(e))