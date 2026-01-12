import joblib

PIPE_STEP1 = "models/model_stage1.pkl"
PIPE_STEP2 = "models/model_stage2.pkl"

pipe_step1 = joblib.load(PIPE_STEP1)
pipe_step2 = joblib.load(PIPE_STEP2)

def predict_eye(X):
    # Étape 1 : Normal vs KC
    p_kc = pipe_step1.predict_proba(X)[0][1]

    if p_kc >= 0.5:
        return {
            "diagnosis": "Kératocône avéré",
            "probability": round(p_kc, 3)
        }

    # Étape 2 : Normal vs Fruste
    p_fruste = pipe_step2.predict_proba(X)[0][1]

    if p_fruste >= 0.6:
        return {
            "diagnosis": "Kératocône fruste (suspect)",
            "probability": round(p_fruste, 3)
        }

    return {
        "diagnosis": "Normal",
        "probability": round(1 - p_fruste, 3)
    }
