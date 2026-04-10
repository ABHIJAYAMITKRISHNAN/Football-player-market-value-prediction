import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.express as px
from frontend.components.styles import apply_custom_styles
from frontend.components.charts import apply_theme
from src.preprocessing import load_and_preprocess
from src.utils import INJURY_FEATURES

st.set_page_config(page_title="Injury Risk Tracker", layout="wide")
apply_custom_styles()

@st.cache_resource
def load_injury_model():
    return joblib.load('saved_models/injury_classifier.pkl')

@st.cache_data
def get_data():
    return load_and_preprocess('data/football.csv')

df = get_data()
model = load_injury_model()

st.markdown('<div class="section-header">🏥 Injury Risk Classification</div>', unsafe_allow_html=True)

player_name = st.selectbox("Select Player", options=[""] + sorted(df['name'].tolist()))

if player_name:
    row = df[df['name'] == player_name].iloc[0]
    
    # Predict
    X_input = pd.DataFrame([row[INJURY_FEATURES]])
    probs = model.predict_proba(X_input)[0]
    classes = model.classes_ # Low, Medium, High usually
    pred_idx = np.argmax(probs)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        risk_label = classes[pred_idx]
        badge_map = {'Low': 'badge-low', 'Medium': 'badge-medium', 'High': 'badge-high'}
        st.markdown(f'<div class="badge {badge_map.get(risk_label, "badge-fair")}">{risk_label} Risk</div>', unsafe_allow_html=True)
        
        st.write("")
        for cls, prob in zip(classes, probs):
            st.write(f"**{cls}**")
            st.progress(float(prob))
            
    with col2:
        st.subheader("Injury Drivers (SHAP)")
        explainer = shap.TreeExplainer(model)
        # For multiclass RF, shap_values is a list. We show for the predicted class.
        shap_values = explainer.shap_values(X_input)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        # Note: Depending on sklearn version and SHAP, indexing might differ
        idx_in_classes = list(classes).index(risk_label)
        
        shap.plots.waterfall(shap.Explanation(
            values=shap_values[:,:,idx_in_classes][0] if hasattr(shap_values, 'shape') and len(shap_values.shape)==3 else shap_values[idx_in_classes][0], 
            base_values=explainer.expected_value[idx_in_classes], 
            data=X_input.iloc[0], 
            feature_names=INJURY_FEATURES
        ), show=False)
        st.pyplot(fig)
        plt.clf()

st.markdown("---")
st.subheader("Global Injury Landscape")
fig = px.scatter(df, x='age', y='days_injured', color='injury_risk_label', 
                 hover_name='name', color_discrete_map={'Low': '#00ff87', 'Medium': '#d29922', 'High': '#f85149'})
st.plotly_chart(apply_theme(fig), use_container_width=True)
