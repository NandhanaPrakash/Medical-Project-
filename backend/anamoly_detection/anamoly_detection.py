import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib

# === Step 1: Load and Clean Data ===
df_raw = pd.read_excel("anamoly.xlsx", header=2)

df_raw.columns = [
    'Age', 'BGL', 'Diastolic_BP', 'Systolic_BP',
    'Heart_Rate', 'Body_Temperature', 'SpO2',
    'Sweating', 'Shivering', 'Diabetic_Status'
]

df = df_raw[pd.to_numeric(df_raw['BGL'], errors='coerce').notnull()].copy()

for col in ['Age', 'BGL', 'Diastolic_BP', 'Systolic_BP', 'Heart_Rate', 'Body_Temperature', 'SpO2', 'Sweating', 'Shivering']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# === Step 2: Feature Selection (with symptoms and age) ===
features = ['Age', 'BGL', 'Diastolic_BP', 'Systolic_BP', 'Heart_Rate', 'Body_Temperature', 'SpO2', 'Sweating', 'Shivering']
X = df[features].dropna()

# === Step 3: Train Isolation Forest Model ===
model = IsolationForest(n_estimators=150, contamination=0.12, random_state=42)
df.loc[X.index, 'anomaly_label'] = model.fit_predict(X)

# Save the model
joblib.dump(model, 'anomaly_detector_model_age_aware.joblib')

# === Step 4: Age-aware anomaly interpretation ===
def interpret_row(row):
    if row['anomaly_label'] == 1:
        return "Normal"

    reasons = []

    # === Age Group Thresholds ===
    age = row['Age']
    is_child = age < 18
    is_senior = age >= 60

    # Age-aware BP
    if (row['Systolic_BP'] > 130 and not is_child) or (row['Systolic_BP'] > 120 and is_child):
        reasons.append("High Systolic BP")
    if (row['Diastolic_BP'] > 85 and not is_child) or (row['Diastolic_BP'] > 75 and is_child):
        reasons.append("High Diastolic BP")
    if (row['Systolic_BP'] < 90):
        reasons.append("Low Systolic BP")
    if (row['Diastolic_BP'] < 60):
        reasons.append("Low Diastolic BP")

    # Age-aware Heart Rate
    if (is_child and row['Heart_Rate'] > 120) or (is_senior and row['Heart_Rate'] > 100) or (not is_child and not is_senior and row['Heart_Rate'] > 105):
        reasons.append("High Heart Rate")
    elif row['Heart_Rate'] < 60:
        reasons.append("Low Heart Rate")

    # Body Temp
    if row['Body_Temperature'] > 100.4:
        reasons.append("High Body Temperature")
    elif row['Body_Temperature'] < 97.0:
        reasons.append("Low Body Temperature")

    # Oxygen Saturation
    if row['SpO2'] < 95:
        reasons.append("Low SpO2")

    # Symptoms
    if row['Sweating'] == 1:
        reasons.append("Sweating")
    if row['Shivering'] == 1:
        reasons.append("Shivering")

    return ", ".join(reasons) if reasons else "Unclassified Anomaly"

df.loc[X.index, 'anomaly_description'] = df.loc[X.index].apply(interpret_row, axis=1)

# === Step 5: Save Results ===
df.to_csv("anomaly_detected_age_aware.csv", index=False)

print("✅ Age-aware anomaly detection complete.")
print("📁 Model saved as: anomaly_detector_model_age_aware.joblib")
print("📄 Results saved in: anomaly_detected_age_aware.csv")
