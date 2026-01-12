import joblib

PIPE_STEP1 = "models/model_stage1.pkl"
PIPE_STEP2 = "models/model_stage2.pkl"

pipe_step1 = joblib.load(PIPE_STEP1)
pipe_step2 = joblib.load(PIPE_STEP2)

def predict_eye(df_prepared):
    results_list = []
    
    # On boucle sur chaque ligne (chaque œil) du DataFrame
    for i in range(len(df_prepared)):
        # Extraction de la ligne i sous forme de DataFrame (pour garder les noms de colonnes)
        X = df_prepared.iloc[[i]]
        
        # Étape 1 : Probabilité KC vs Normal
        p_kc_brut = pipe_step1.predict_proba(X)[0][1] 
        p_not_kc = 1 - p_kc_brut

        # Étape 2 : Probabilité Fruste vs Normal
        p_fruste_dans_suspect = pipe_step2.predict_proba(X)[0][1]

        # Calcul final
        prob_kc = p_kc_brut
        prob_fruste = p_not_kc * p_fruste_dans_suspect
        prob_normal = p_not_kc * (1 - p_fruste_dans_suspect)

        probs = {
            "Kératocône avéré": round(prob_kc, 3),
            "Kératocône fruste": round(prob_fruste, 3),
            "Normal": round(prob_normal, 3)
        }

        dominant_class = max(probs, key=probs.get)
        
        results_list.append({
            "eye_index": i + 1,
            "prediction": dominant_class,
            "all_probs": probs
        })
        
    return results_list
