import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from frontend.components.styles import apply_custom_styles, render_kpi
from src.preprocessing import load_and_preprocess
from src.utils import BASE_FEATURES

st.set_page_config(page_title="Price Prediction", layout="wide")
apply_custom_styles()

@st.cache_resource
def load_models():
    model = joblib.load('saved_models/price_predictor.pkl')
    scaler = joblib.load('saved_models/scaler.pkl')
    return model, scaler

@st.cache_data
def get_data():
    return load_and_preprocess('data/football.csv')

df = get_data()
model, scaler = load_models()

st.markdown('<div class="section-header">💰 Market Value Predictor</div>', unsafe_allow_html=True)

mode = st.radio("Mode", ["Search Existing Player", "Manual Input"], horizontal=True)

input_data = {}

if mode == "Search Existing Player":
    player_name = st.selectbox("Select Player", options=[""] + sorted(df['name'].tolist()))
    if player_name:
        row = df[df['name'] == player_name].iloc[0]
        for feat in BASE_FEATURES:
            input_data[feat] = row[feat]
else:
    cols = st.columns(3)
    for i, feat in enumerate(BASE_FEATURES):
        with cols[i % 3]:
            # Use appropriate step/range based on feature name
            if 'age' in feat: input_data[feat] = st.slider(feat, 15, 45, 25)
            elif 'height' in feat: input_data[feat] = st.slider(feat, 150, 210, 180)
            elif 'minutes' in feat: input_data[feat] = st.number_input(feat, 0, 10000, 2000)
            else: input_data[feat] = st.number_input(feat, 0.0, 1000.0, 1.0)

if st.button("Predict Market Value"):
    X_input = pd.DataFrame([input_data])[BASE_FEATURES]
    X_scaled = scaler.transform(X_input)
    
    pred_log = model.predict(X_scaled)[0]
    pred_val = np.expm1(pred_log)
    
    st.markdown("---")
    c1, c2 = st.columns([1, 2])
    
    with c1:
        render_kpi("Predicted Value", f"€{pred_val:,.0f}")
        st.caption("Confidence Range (±15%):")
        st.write(f"€{pred_val*0.85:,.0f} - €{pred_val*1.15:,.0f}")
        
    with c2:
        st.subheader("Value Drivers (SHAP)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots.waterfall(shap.Explanation(
            values=shap_values[0], 
            base_values=explainer.expected_value, 
            data=X_input.iloc[0], 
            feature_names=BASE_FEATURES
        ), show=False)
        st.pyplot(fig)
        plt.clf()
