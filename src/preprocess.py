# Convert raw UNSW-NB15 data into preprocessed machine-ready format
import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler

RAW_DIR    = Path(__file__).resolve().parents[1] / "data" / "raw"
SAVE_DIR   = Path(__file__).resolve().parents[1] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

CATEGORICAL_COLS = ['proto', 'service', 'state']

def preprocess_data():
    """Load raw data, clean, encode, scale, and save to data/processed/."""
    print("Loading datasets...")
    train = pd.read_parquet(RAW_DIR / "UNSW_NB15_training-set.parquet")
    test  = pd.read_parquet(RAW_DIR / "UNSW_NB15_testing-set.parquet")

    print("Separating labels...")
    y_train          = train['label']
    y_test           = test['label']
    attack_cat_train = train['attack_cat']
    attack_cat_test  = test['attack_cat']
    X_train = train.drop(['label', 'attack_cat'], axis=1)
    X_test  = test.drop(['label', 'attack_cat'], axis=1)

    print("Handling missing values...")
    for col in X_train.columns:
        if X_train[col].dtype in ['int64', 'float64']:
            median = X_train[col].median()
            X_train[col] = X_train[col].fillna(median)
            X_test[col]  = X_test[col].fillna(median)
        else:
            mode = X_train[col].mode()[0]
            X_train[col] = X_train[col].fillna(mode)
            X_test[col]  = X_test[col].fillna(mode)

    print("Encoding categorical features...")
    encoders = {}
    for col in CATEGORICAL_COLS:
        encoder = LabelEncoder()
        encoder.fit(pd.concat([X_train[col], X_test[col]]))
        X_train[col] = encoder.transform(X_train[col])
        X_test[col]  = encoder.transform(X_test[col])
        encoders[col] = encoder

    print("Scaling features...")
    scaler = StandardScaler()
    X_train[X_train.columns] = scaler.fit_transform(X_train)
    X_test[X_test.columns]   = scaler.transform(X_test)

    print("Saving preprocessed datasets...")
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(SAVE_DIR / "X_train_preprocessed.parquet")
    X_test.to_parquet(SAVE_DIR  / "X_test_preprocessed.parquet")
    y_train.to_frame().to_parquet(SAVE_DIR         / "y_train.parquet")
    y_test.to_frame().to_parquet(SAVE_DIR           / "y_test.parquet")
    attack_cat_train.to_frame().to_parquet(SAVE_DIR / "attack_cat_train.parquet")
    attack_cat_test.to_frame().to_parquet(SAVE_DIR  / "attack_cat_test.parquet")

    # Save encoders and scaler so dashboard can reuse them
    print("Saving encoders and scaler...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoders, MODELS_DIR / "encoders.pkl")
    joblib.dump(scaler,   MODELS_DIR / "scaler.pkl")

    print("\n✅ Preprocessing complete!")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

if __name__ == "__main__":
    preprocess_data()