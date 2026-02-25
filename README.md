IDS — UNSW-NB15

A network intrusion detection system built using the UNSW-NB15 dataset. It uses Random Forest classifiers to detect whether network traffic is normal or malicious, and if malicious, identifies the attack type.

What it does

Classifies network traffic as normal or an attack (86.91% accuracy)
Identifies the attack category — Generic, Exploits, Fuzzers, DoS, Reconnaissance, Shellcode, Backdoor, Worms, Analysis
Includes a Streamlit dashboard where you can upload a traffic CSV and get predictions in real time

Features

- Random Forest models for both binary and multi-class classification
- Attack categorization across 9 attack types — Generic, Exploits, Fuzzers, DoS, Reconnaissance, Shellcode, Backdoor, Worms, Analysis
- Feature importance analysis with a reduced 15-feature model for faster inference
- Streamlit dashboard — upload a CSV, get predictions, risk scores, and attack type breakdown in real time
- Batch prediction with per-record threat reporting and severity scoring
- Logs all predictions to CSV for review

Dataset
UNSW-NB15 — created by the Australian Centre for Cyber Security. Contains over 2 million network traffic records across 9 attack categories.

Screenshots:

Pic01-RESULTS

<img width="570" height="838" alt="image" src="https://github.com/user-attachments/assets/a3b9e74a-62af-49a9-bd7f-639c7d686bda" />

Pic02.1-DASHBOARD

<img width="1847" height="953" alt="image" src="https://github.com/user-attachments/assets/a0b9ef7e-36b7-4d0f-9b6f-95cef964ed9a" />


Pic02.2-DASHBOARD

<img width="1797" height="810" alt="image" src="https://github.com/user-attachments/assets/bc176056-ed6e-457e-80a1-6988a57ddd61" />
