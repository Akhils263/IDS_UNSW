# preprocess_upload.py — Preprocess a user-uploaded CSV for inference
import joblib
import pandas as pd
from pathlib import Path

MODELS_DIR       = Path(__file__).resolve().parents[1] / "models"
CATEGORICAL_COLS = ['proto', 'service', 'state']

def preprocess_uploaded(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same encoding and scaling used during training to an uploaded CSV.

    The uploaded CSV should contain raw network traffic features (no label or
    attack_cat columns). The encoders and scaler saved during training are
    loaded and applied so the data is compatible with the trained model.

    Parameters
    ----------
    df : pd.DataFrame
        Raw network traffic data uploaded by the user.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame ready for model inference.
    """
    df = df.copy()

    # Drop label columns if user accidentally includes them
    df = df.drop(columns=[c for c in ['label', 'attack_cat'] if c in df.columns])

    # Load encoders and scaler saved during training
    encoders = joblib.load(MODELS_DIR / "encoders.pkl")
    scaler   = joblib.load(MODELS_DIR / "scaler.pkl")

    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categorical columns
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            encoder = encoders[col]
            # Handle unseen categories gracefully
            df[col] = df[col].apply(
                lambda x: x if x in encoder.classes_ else encoder.classes_[0]
            )
            df[col] = encoder.transform(df[col])

    # Scale
    df[df.columns] = scaler.transform(df)

    return df