import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from load_data import load_preprocessed_data

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

def train_binary_model(X_train, y_train):
    """Train a Random Forest for binary classification (normal vs attack)."""
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def train_attack_model(X_train, attack_cat_train):
    """Train a Random Forest for multi-class attack category classification."""
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, attack_cat_train)
    return model

def evaluate_model(model, X_test, y_test, label=""):
    """Print confusion matrix and classification report for a given model."""
    y_pred = model.predict(X_test)
    if label:
        print(f"\n--- {label} ---")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

def save_models(rf_binary, rf_attack):
    """Save trained models to the models/ directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_binary, MODELS_DIR / "rf_binary.pkl")
    joblib.dump(rf_attack, MODELS_DIR / "rf_attack_category.pkl")
    print("\n✅ Models saved:")
    print(f"  - {MODELS_DIR / 'rf_binary.pkl'}")
    print(f"  - {MODELS_DIR / 'rf_attack_category.pkl'}")

def main():
    print("Loading preprocessed data...")
    X_train, X_test, y_train, y_test, attack_cat_train, attack_cat_test = load_preprocessed_data()

    print("\nTraining binary model (normal vs attack)...")
    rf_binary = train_binary_model(X_train, y_train)
    evaluate_model(rf_binary, X_test, y_test, label="Binary Classification")

    print("\nTraining attack category model...")
    rf_attack = train_attack_model(X_train, attack_cat_train)
    evaluate_model(rf_attack, X_test, attack_cat_test, label="Attack Category Classification")

    save_models(rf_binary, rf_attack)

if __name__ == "__main__":
    main()