import pandas as pd
import numpy as np
import joblib
import json
#from chatbot_connector import send_to_chatbot  # Chatbot integration function

# -------------------- Data Cleaning --------------------
def clean_data(df):
    df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True).str.replace(r'[^\w\s]', '', regex=True)
    
    rename_map = {
        'Blood_Glucose_LevelBGL': 'BGL',
        'Diastolic_Blood_Pressure': 'Diastolic_BP',
        'Systolic_Blood_Pressure': 'Systolic_BP',
        'Body_Temperature': 'Body_Temperature',
        'Heart_Rate': 'Heart_Rate',
        'Sweating_(YN)': 'Sweating',
        'Shivering_(YN)': 'Shivering',
        'Sweating_YN': 'Sweating',
        'Shivering_YN': 'Shivering',
        'SPO2': 'SpO2'
    }
    
    df.rename(columns=rename_map, inplace=True)
    
    # Convert 'Y'/'N'/1/0 to consistent binary values
    df['Sweating'] = df['Sweating'].astype(str).str.upper().map({'Y': 1, 'N': 0, '1': 1, '0': 0})
    df['Shivering'] = df['Shivering'].astype(str).str.upper().map({'Y': 1, 'N': 0, '1': 1, '0': 0})
    
    return df

# -------------------- Anomaly Detection Pipeline --------------------
def detect_anomalies_and_send(data_path: str):
    # Load and preprocess input data
    df = pd.read_excel(data_path, header=2)
    df = clean_data(df)

    # Define relevant features for anomaly detection
    features = ['Age', 'BGL', 'Diastolic_BP', 'Systolic_BP', 'Heart_Rate',
                'Body_Temperature', 'SpO2', 'Sweating', 'Shivering']
    
    if not all(f in df.columns for f in features):
        raise ValueError("❌ Required features are missing from the input data.")
    
    X = df[features]

    # Load the trained Isolation Forest model
    try:
        model = joblib.load("anomaly_detector.joblib")
        print("✅ Loaded Isolation Forest model.")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model: {e}")


    # Predict anomalies (-1 for anomaly, 1 for normal)
    predictions = model.predict(X)
    df['Anomaly_Prediction'] = predictions

    # Filter anomaly rows
    anomalies = df[df['Anomaly_Prediction'] == -1]

    if anomalies.empty:
        print("✅ No anomalies detected.")
        return

    print(f"🚨 {len(anomalies)} anomaly/anomalies detected. Sending to chatbot...\n")

    # Send each anomaly to chatbot
    for index, row in anomalies.iterrows():
        anomaly_info = {
            "timestamp": str(row.get("Timestamp", "N/A")),
            "age": row['Age'],
            "BGL": row['BGL'],
            "BP": f"{row['Systolic_BP']}/{row['Diastolic_BP']}",
            "Heart_Rate": row['Heart_Rate'],
            "SpO2": row['SpO2'],
            "Temperature": row['Body_Temperature'],
            "Sweating": bool(row['Sweating']),
            "Shivering": bool(row['Shivering']),
            "message": "Potential health anomaly detected in patient vitals."
        }

        try:
            response = send_to_chatbot(anomaly_info)
            print("🧠 Chatbot Insight:")
            print(json.dumps(response, indent=2))
        except Exception as e:
            print(f"❌ Error sending to chatbot: {e}")

# -------------------- Entry Point --------------------
if __name__ == "__main__":
    detect_anomalies_and_send("anamoly.xlsx")
