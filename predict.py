import joblib

PIPE_STEP1 = "models/model_stage1.pkl"
PIPE_STEP2 = "models/model_stage2.pkl"

pipe_step1 = joblib.load(PIPE_STEP1)
pipe_step2 = joblib.load(PIPE_STEP2)

def predict_eye(X):
    # Étape 1 : Probabilité d'être atteint (KC ou Fruste) vs Normal
    # On récupère la probabilité de la classe "Malade/KC"
    p_kc_brut = pipe_step1.predict_proba(X)[0][1] 
    p_not_kc = 1 - p_kc_brut

    # Étape 2 : Si ce n'est pas un KC franc, quelle est la probabilité Fruste vs Normal
    p_fruste_dans_suspect = pipe_step2.predict_proba(X)[0][1]

    # Calcul des probabilités finales combinées
    prob_kc = p_kc_brut
    prob_fruste = p_not_kc * p_fruste_dans_suspect
    prob_normal = p_not_kc * (1 - p_fruste_dans_suspect)

    # On crée un dictionnaire des résultats
    results = {
        "Kératocône avéré": round(prob_kc, 3),
        "Kératocône fruste": round(prob_fruste, 3),
        "Normal": round(prob_normal, 3)
    }

    # Trouver la classe dominante
    prediction = max(results, key=results.get)

    return {
        "prediction": prediction,
        "all_probs": results
    }
