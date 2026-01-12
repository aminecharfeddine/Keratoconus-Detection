import pandas as pd
import joblib

FEATURES_PATH = "models/feature_names.pkl"

def load_feature_names():
    return joblib.load(FEATURES_PATH)

def preprocess_eye(df_raw):
    df = df_raw.copy()

    # 1. Supprimer colonnes non utilisées
    drop_cols = [
        "PatientID", "Last Name", "First Name", "DOB", "Age",
        "Gender", "Ethnicity", "Eye",
        "Scan Date", "Scan Time"
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 2. Supprimer colonnes vides / parasites
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # 3. Aligner EXACTEMENT les features d'entraînement
    expected_features = load_feature_names()
    df = df.reindex(columns=expected_features)

    return df
