import pandas as pd
from pathlib import Path

# Project root = one level up from src/
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

def load_preprocessed_data():
    """Load preprocessed UNSW-NB15 datasets from parquet files."""
    X_train          = pd.read_parquet(DATA_DIR / "X_train_preprocessed.parquet")
    X_test           = pd.read_parquet(DATA_DIR / "X_test_preprocessed.parquet")
    y_train          = pd.read_parquet(DATA_DIR / "y_train.parquet").squeeze("columns")
    y_test           = pd.read_parquet(DATA_DIR / "y_test.parquet").squeeze("columns")
    attack_cat_train = pd.read_parquet(DATA_DIR / "attack_cat_train.parquet").squeeze("columns")
    attack_cat_test  = pd.read_parquet(DATA_DIR / "attack_cat_test.parquet").squeeze("columns")
    return X_train, X_test, y_train, y_test, attack_cat_train, attack_cat_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, attack_cat_train, attack_cat_test = load_preprocessed_data()
    print("✅ Data loaded successfully!")
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train distribution:\n", y_train.value_counts())
    print("Attack categories:\n", attack_cat_train.value_counts())