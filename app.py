from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route("/", methods=["GET"])
def index():
    return "Crop Recommendation API Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    pred = model.predict(df)[0]
    return {"crop": pred}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
