# inference.py — Run evaluation on the full test set
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from load_data import load_preprocessed_data

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

print("🔄 Loading preprocessed data...")
X_train, X_test, y_train, y_test, attack_cat_train, attack_cat_test = load_preprocessed_data()

print("🔄 Loading trained model...")
model = joblib.load(MODELS_DIR / "rf_binary.pkl")
print("✅ Model loaded successfully!")

print("\n🔎 Running inference on the entire test dataset...")
y_pred = model.predict(X_test)

print("\n📊 Binary Classification Results (Normal vs Attack):")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

try:
    attack_model = joblib.load(MODELS_DIR / "rf_attack_category.pkl")
    attack_pred  = attack_model.predict(X_test)
    print("\n📊 Multi-class Classification Results (Attack Categories):")
    print(classification_report(attack_cat_test, attack_pred, digits=4))
except FileNotFoundError:
    print("\n⚠️ Attack model not found. Skipping attack type evaluation.")

print("\n✅ Inference finished successfully!")
