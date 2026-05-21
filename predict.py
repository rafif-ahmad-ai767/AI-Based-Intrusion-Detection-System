import joblib
import numpy as np

model = joblib.load("models/ids_model.pkl")
scaler = joblib.load("models/scaler.pkl")

def predict_intrusion(sample):
    sample = scaler.transform([sample])
    prediction = model.predict(sample)
    return prediction[0]
