import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/ids_model.pkl")
scaler = joblib.load("models/scaler.pkl")

st.title("AI-Based Intrusion Detection System")

uploaded = st.file_uploader("Upload Network Traffic CSV")

if uploaded:
    df = pd.read_csv(uploaded)
    df_scaled = scaler.transform(df)
    preds = model.predict(df_scaled)

    df['Prediction'] = preds
    st.dataframe(df)

    st.bar_chart(df['Prediction'].value_counts())
