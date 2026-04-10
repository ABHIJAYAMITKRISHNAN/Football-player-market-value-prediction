import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from frontend.components.styles import apply_custom_styles
from frontend.components.charts import apply_theme
from src.preprocessing import load_and_preprocess

st.set_page_config(page_title="Value Trajectory", layout="wide")
apply_custom_styles()

@st.cache_resource
def load_dep_model():
    return joblib.load('saved_models/depreciation_model.pkl')

@st.cache_data
def get_data():
    return load_and_preprocess('data/football.csv')

df = get_data()
model = load_dep_model()

st.markdown('<div class="section-header">📉 Value Depreciation / Appreciation</div>', unsafe_allow_html=True)

player_name = st.selectbox("Select Player", options=[""] + sorted(df['name'].tolist()))

if player_name:
    row = df[df['name'] == player_name].iloc[0]
    
    # Predict
    dep_features = [
        'age', 'age_group_encoded', 'appearance', 'injury_burden', 
        'days_injured', 'productivity_score', 'award', 'position_encoded'
    ]
    X_input = pd.DataFrame([row[dep_features]])
    probs = model.predict_proba(X_input)[0]
    classes = ['Appreciating', 'Stable', 'Depreciating']
    pred_idx = np.argmax(probs)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        badge_class = ["badge-low", "badge-fair", "badge-high"][pred_idx]
        st.markdown(f'<div class="badge {badge_class}">{classes[pred_idx]}</div>', unsafe_allow_html=True)
        
        st.write("")
        for i, (cls, prob) in enumerate(zip(classes, probs)):
            st.write(f"**{cls}**")
            st.progress(float(prob))
            
    with col2:
        st.subheader("Value Trend by Age (Same Position)")
        group_df = df[df['position_group'] == row['position_group']]
        age_trend = group_df.groupby('age')['current_value'].mean().reset_index()
        
        fig = px.line(age_trend, x='age', y='current_value', title=f"Avg Market Value for {row['position_group']}", 
                      color_discrete_sequence=['#00ff87'])
        fig.add_scatter(x=[row['age']], y=[row['current_value']], mode='markers+text', 
                         name=row['name'], text=[row['name']], textposition="top center",
                         marker=dict(size=12, color='#ffd700'))
        st.plotly_chart(apply_theme(fig), use_container_width=True)
