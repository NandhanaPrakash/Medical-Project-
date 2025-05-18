import requests
import pandas as pd

CHATBOT_API_URL = "http://localhost:8000/chat"

def send_to_chatbot(anomaly_data):
    try:
        response = requests.post(CHATBOT_API_URL, json=anomaly_data)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to connect to chatbot: {str(e)}"}

if __name__ == "__main__":
    # Load the anomaly data
    df = pd.read_excel("anamoly_detection/anomaly_detection_results.xlsx")

    # Loop through each row and send the description to the chatbot
    for idx, row in df.iterrows():
        label = row['anomaly_label']
        description = row['anomaly_description']

        print(f"\n📌 Patient {idx+1} - Label: {label}")
        result = send_to_chatbot({"message": description})
        print("🩺 Chatbot Response:", result.get("insight") or result.get("error"))
