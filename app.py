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

uploaded_file = st.file_uploader(
    "Importer un fichier de topographie (.txt)",
    type=["txt", "csv"]
)

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file, sep="\t", engine="python")
    st.success(f"Fichier importé : {len(df_raw)} œil/yeux détecté(s)")

    df_prepared = preprocess_eye(df_raw)
    results = predict_eye(df_prepared) # C'est maintenant une liste

    # Création d'onglets pour chaque œil
    tabs = st.tabs([f"Œil {res['eye_index']}" for res in results])

    for i, res in enumerate(results):
        with tabs[i]:
            dominant_class = res["prediction"]
            all_probs = res["all_probs"]

            st.metric("Diagnostic suggéré", dominant_class)
            
            # Affichage des probabilités détaillées
            for label, prob in all_probs.items():
                if label == dominant_class:
                    st.write(f"**➡️ {label} : {prob*100:.1f}%**")
                else:
                    st.write(f"{label} : {prob*100:.1f}%")
