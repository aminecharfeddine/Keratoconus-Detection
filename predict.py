import joblib
import numpy as np

MODEL_STEP1 = "models/model_stage1.pkl"
MODEL_STEP2 = "models/model_stage2.pkl"
SCALER_PATH = "models/scaler.pkl"

model_step1 = joblib.load(MODEL_STEP1)
model_step2 = joblib.load(MODEL_STEP2)
scaler = joblib.load(SCALER_PATH)

def predict_eye(X):
    X_scaled = scaler.transform(X)

    # Étape 1 : Normal vs KC
    p_kc = model_step1.predict_proba(X_scaled)[0][1]

    if p_kc > 0.5:
        return {
            "diagnosis": "Kératocône avéré",
            "probability": round(p_kc, 3)
        }

    # Étape 2 : Normal vs Fruste
    p_fruste = model_step2.predict_proba(X_scaled)[0][1]

    if p_fruste > 0.5:
        return {
            "diagnosis": "Kératocône fruste (suspect)",
            "probability": round(p_fruste, 3)
        }

    return {
        "diagnosis": "Normal",
        "probability": round(1 - p_fruste, 3)
    }
