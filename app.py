import streamlit as st
import pandas as pd
from preprocessing import preprocess_eye
from predict import predict_eye

st.set_page_config(
    page_title="Aide à la détection du kératocône",
    layout="centered"
)

st.title("🩺 Détection du kératocône")
st.markdown("**Outil d’aide à la décision – usage non diagnostique**")

uploaded_file = st.file_uploader("Importer un fichier (.txt)", type=["txt", "csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file, sep="\t", engine="python")
    
    # 1. On récupère les labels AVANT le preprocessing (car le preprocessing risque de supprimer la colonne 'Eye')
    eye_labels = None
    if 'Eye' in df_raw.columns:
        eye_labels = df_raw['Eye'].tolist()

    st.success(f"Fichier importé avec succès ({len(df_raw)} œil/yeux)")

    # 2. Calculs
    df_prepared = preprocess_eye(df_raw)
    results = predict_eye(df_prepared, labels=eye_labels)

    # 3. Affichage avec onglets personnalisés
    tabs = st.tabs([res['eye_label'] for res in results])

    for i, res in enumerate(results):
        with tabs[i]:
            dominant_class = res["prediction"]
            all_probs = res["all_probs"]

            st.subheader(f"Analyse pour l'{res['eye_label']}")
            
            # Affichage du résultat principal
            st.metric("Diagnostic suggéré", dominant_class)
            
            # Détail des probabilités
            st.write("Détails des probabilités :")
            for label, prob in all_probs.items():
                if label == dominant_class:
                    # Mise en gras et couleur pour le plus probable
                    st.markdown(f"**➡️ {label} : {prob*100:.1f}% (Confiance maximale)**")
                else:
                    st.write(f"{label} : {prob*100:.1f}%")
            
            # Petit conseil visuel
            if dominant_class == "Kératocône fruste" and all_probs["Normal"] > 0.30:
                st.info("💡 Note : Le score de normalité est significatif. Cas à surveiller de près.")

    st.warning("⚠️ Usage réservé à l'aide à la décision clinique.")
