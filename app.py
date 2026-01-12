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
    st.success("Fichier importé avec succès")

    # Calcul
    df_prepared = preprocess_eye(df_raw)
    result = predict_eye(df_prepared)
    
    st.subheader("Analyse des probabilités")
    
    # Affichage des résultats
    dominant_class = result["prediction"]
    all_probs = result["all_probs"]

    for label, prob in all_probs.items():
        # Si c'est la classe prédite, on met en gras et on ajoute une icône
        if label == dominant_class:
            st.markdown(f"**➡️ {label} : {prob*100:.1f}% (Le plus probable)**")
        else:
            st.write(f"{label} : {prob*100:.1f}%")

    # Rappel visuel
    st.info(f"Interprétation suggérée : **{dominant_class}**")

    st.warning(
        "⚠️ Cet outil est une aide à la décision et ne remplace pas un examen clinique complet."
    )
