import joblib
import time
from pathlib import Path
from datetime import datetime
from load_data import load_preprocessed_data

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
LOGS_DIR   = Path(__file__).resolve().parents[1] / "logs"
STATE_FILE = Path(__file__).resolve().parents[1] / "system_state.txt"

# --- Attack descriptions ---
ATTACK_DESCRIPTIONS = {
    "DoS":            "Denial of Service — traffic flooding attempt",
    "Fuzzers":        "Malformed packet attack",
    "Exploits":       "System vulnerability exploitation attempt",
    "Reconnaissance": "Network scanning / probing",
    "Generic":        "Brute force / general intrusion",
    "Shellcode":      "Code injection attempt",
    "Worms":          "Self-propagating malware activity"
}

def calculate_severity(risk_score):
    if risk_score < 30:   return "Low"
    elif risk_score < 60: return "Medium"
    elif risk_score < 80: return "High"
    else:                 return "Critical"

def load_models():
    binary_model = joblib.load(MODELS_DIR / "rf_binary.pkl")
    attack_model = joblib.load(MODELS_DIR / "rf_attack_category.pkl")
    return binary_model, attack_model

def load_state():
    state = {}
    with open(STATE_FILE, "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            state[key] = int(value)
    return state

def save_state(state):
    with open(STATE_FILE, "w") as f:
        for key, value in state.items():
            f.write(f"{key}={value}\n")

def predict_main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data and models...")
    X_train, X_test, y_train, y_test, attack_cat_train, attack_cat_test = load_preprocessed_data()
    binary_model, attack_model = load_models()
    state = load_state()

    # Grab next batch of 10 records
    chunk_size = 10
    start  = state["chunk_start"]
    sample = X_test.iloc[start:start + chunk_size]

    start_time = time.time()
    pred  = binary_model.predict(sample)
    probs = binary_model.predict_proba(sample)

    print("\n🚨 ===== THREAT ANALYSIS REPORT =====\n")
    attack_records = []

    for i in range(len(pred)):
        label      = "Attack" if pred[i] == 1 else "Normal"
        confidence = round(max(probs[i]) * 100, 2)
        risk_score = probs[i][1] * 100
        severity   = calculate_severity(risk_score)
        state["event_id"] += 1
        attack_type = "Normal"

        print(f"Record {i+1}:")
        print(f"  Type:       {label}")
        print(f"  Severity:   {severity}")
        print(f"  Risk Score: {round(risk_score, 2)}")
        print(f"  Confidence: {confidence}%")

        if label == "Attack":
            attack_type = attack_model.predict(sample.iloc[[i]])[0]
            attack_records.append(risk_score)
            print(f"  Attack Category: {attack_type}")
            print(f"  Description:     {ATTACK_DESCRIPTIONS.get(attack_type, 'Unknown threat pattern')}")
            print("  Action:          Monitor or block suspicious traffic")

        # Log individual event
        with open(LOGS_DIR / "attack_events.csv", "a") as f:
            f.write(f"{datetime.now()},{state['event_id']},{label},{severity},{risk_score:.2f},{round(probs[i][1],2)},{attack_type}\n")

        print()

    # Update state for next batch
    state["chunk_start"] = start + chunk_size
    save_state(state)

    elapsed = round(time.time() - start_time, 4)
    total   = len(pred)
    attacks = sum(pred)
    normal  = total - attacks

    print(f"⏱  Batch Detection Time: {elapsed} sec")
    print("\n📊 ===== TRAFFIC SUMMARY =====")
    print(f"  Total Records:    {total}")
    print(f"  Normal Traffic:   {normal}")
    print(f"  Attacks Detected: {attacks}")
    if attack_records:
        print(f"  Avg Risk Score:   {round(sum(attack_records)/len(attack_records), 2)}")

    # Log batch summary
    with open(LOGS_DIR / "ids_report_log.csv", "a") as f:
        f.write(f"{datetime.now()},Total:{total},Attacks:{attacks},Normal:{normal}\n")

if __name__ == "__main__":
    predict_main()