# inference_selected.py — Run evaluation using the selected-features model
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from load_data import load_preprocessed_data

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

SELECTED_FEATURES = [
    "dload", "rate", "ackdat", "tcprtt", "synack",
    "dmean", "sload", "dur", "sinpkt", "sbytes",
    "smean", "ct_dst_sport_ltm", "ct_src_dport_ltm",
    "service", "dbytes"
]

print("Loading trained model...")
model = joblib.load(MODELS_DIR / "rf_selected_features.pkl")

print("Loading preprocessed data...")
X_train, X_test, y_train, y_test, _, _ = load_preprocessed_data()

print("Selecting important features...")
X_test = X_test[SELECTED_FEATURES]

print("Running inference on test set...")
y_pred = model.predict(X_test)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

print("✅ Inference completed successfully")