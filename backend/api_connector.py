from flask import Flask, request, jsonify

app = Flask(__name__)

# -------------------- Dummy Endpoints --------------------
@app.route("/")
def root():
    return jsonify({"message": "Medical Project API running 🎉"})

@app.route("/api/diet-recommendation", methods=["POST"])
def diet_recommendation():
    data = request.get_json()
    # Dummy response
    return jsonify({
        "bmi": 22.5,
        "glucose": 95,
        "recommendation": "Maintain current diet. Consider light exercise."
    })

@app.route("/api/diabetes-prediction", methods=["POST"])
def diabetes_prediction():
    data = request.get_json()
    # Dummy response
    return jsonify({
        "diabetic": True,
        "confidence": 0.87
    })

@app.route("/api/anomaly-detection", methods=["POST"])
def anomaly_detection():
    data = request.get_json()
    # Dummy response
    return jsonify({
        "status": "Anomaly detected",
        "reasons": ["High Heart Rate", "Low SpO2", "Shivering"]
    })

if __name__ == "__main__":
    app.run(debug=True)
