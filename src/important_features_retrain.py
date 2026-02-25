import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from load_data import load_preprocessed_data

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

# Top 15 features selected from feature importance analysis
SELECTED_FEATURES = [
    "dload", "rate", "ackdat", "tcprtt", "synack",
    "dmean", "sload", "dur", "sinpkt", "sbytes",
    "smean", "ct_dst_sport_ltm", "ct_src_dport_ltm",
    "service", "dbytes"
]

def main():
    print("Loading preprocessed data...")
    X_train, X_test, y_train, y_test, _, _ = load_preprocessed_data()

    X_train_sel = X_train[SELECTED_FEATURES]
    X_test_sel  = X_test[SELECTED_FEATURES]

    print("Training Random Forest with selected features...")
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train_sel, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test_sel)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODELS_DIR / "rf_selected_features.pkl")
    print("\n✅ Model saved to:", MODELS_DIR / "rf_selected_features.pkl")

if __name__ == "__main__":
    main()