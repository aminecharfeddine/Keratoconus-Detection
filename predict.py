import joblib

PIPE_STEP1 = "models/model_stage1.pkl"
PIPE_STEP2 = "models/model_stage2.pkl"

pipe_step1 = joblib.load(PIPE_STEP1)
pipe_step2 = joblib.load(PIPE_STEP2)

def predict_eye(df_prepared, labels=None):
    results_list = []
    
    for i in range(len(df_prepared)):
        X = df_prepared.iloc[[i]]
        
        # Logique de prédiction (inchangée)
        p_kc_brut = pipe_step1.predict_proba(X)[0][1] 
        p_not_kc = 1 - p_kc_brut
        p_fruste_dans_suspect = pipe_step2.predict_proba(X)[0][1]

        prob_kc = p_kc_brut
        prob_fruste = p_not_kc * p_fruste_dans_suspect
        prob_normal = p_not_kc * (1 - p_fruste_dans_suspect)

        probs = {
            "Kératocône avéré": round(prob_kc, 3),
            "Kératocône fruste": round(prob_fruste, 3),
            "Normal": round(prob_normal, 3)
        }

        dominant_class = max(probs, key=probs.get)
        
        # On détermine le nom de l'onglet
        eye_name = labels[i] if labels is not None else f"Œil {i+1}"
        if eye_name == "OD": eye_name = "Œil Droit"
        if eye_name == "OS": eye_name = "Œil Gauche"

        results_list.append({
            "eye_label": eye_name,
            "prediction": dominant_class,
            "all_probs": probs
        })
        
    return results_list
