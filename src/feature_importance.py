import joblib
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
DATA_DIR   = Path(__file__).resolve().parents[1] / "data" / "processed"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"

def get_importance_df(model, feature_names):
    """Return a sorted DataFrame of feature importances."""
    return pd.DataFrame({
        "Feature":    feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

def plot_importance(importance_df, title, save_path):
    """Plot and save a horizontal bar chart of top 10 features."""
    top = importance_df.head(10)
    plt.figure(figsize=(10, 5))
    plt.barh(top["Feature"], top["Importance"])
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"  Plot saved to: {save_path}")

def main():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load feature names once
    X_train       = pd.read_parquet(DATA_DIR / "X_train_preprocessed.parquet")
    feature_names = X_train.columns

    # --- Binary model ---
    print("=== Binary IDS Model — Feature Importance ===")
    binary_model  = joblib.load(MODELS_DIR / "rf_binary.pkl")
    binary_imp_df = get_importance_df(binary_model, feature_names)
    print(binary_imp_df.head(10).to_string(index=False))
    plot_importance(binary_imp_df, "Top 10 Features — Binary IDS Model",
                    REPORTS_DIR / "feature_importance_binary.png")

    # --- Attack category model ---
    print("\n=== Attack Category Model — Feature Importance ===")
    attack_model  = joblib.load(MODELS_DIR / "rf_attack_category.pkl")
    attack_imp_df = get_importance_df(attack_model, feature_names)
    print(attack_imp_df.head(10).to_string(index=False))
    plot_importance(attack_imp_df, "Top 10 Features — Attack Category Model",
                    REPORTS_DIR / "feature_importance_attack.png")

if __name__ == "__main__":
    main()