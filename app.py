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
    "Importer un fichier de topographie / tomographie cornéenne (.txt)",
    type=["txt", "csv"]
)

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file, sep="\t", engine="python")

    st.success("Fichier importé avec succès")

    df_prepared = preprocess_eye(df_raw)

    result = predict_eye(df_prepared)

    st.subheader("Résultat")
    st.metric(
        label="Classification proposée",
        value=result["diagnosis"]
    )
    st.write(f"Probabilité estimée : **{result['probability']}**")

    st.warning(
        "⚠️ Cet outil est une aide à la décision et ne remplace pas un avis médical."
    )
