import streamlit as st
import pandas as pd
import plotly.express as px
from frontend.components.styles import apply_custom_styles
from frontend.components.charts import apply_theme
import os

st.set_page_config(page_title="Valuation Detector", layout="wide")
apply_custom_styles()

if not os.path.exists('saved_models/valuation_results.parquet'):
    st.error("Valuation results not found. Please run train_all.py first.")
    st.stop()

df = pd.read_parquet('saved_models/valuation_results.parquet')

st.markdown('<div class="section-header">⚖️ Market Valuation Analysis</div>', unsafe_allow_html=True)

fig = px.scatter(df, x='predicted_value', y='current_value', color='valuation_label',
                 hover_name='name', hover_data=['team', 'value_gap_pct'],
                 color_discrete_map={'Undervalued': '#00ff87', 'Overvalued': '#f85149', 'Fairly Valued': '#8b949e'})

st.plotly_chart(apply_theme(fig), use_container_width=True)

t1, t2 = st.tabs(["💎 Top Undervalued (Bargains)", "🚩 Top Overvalued (Risk)"])

with t1:
    st.table(df[df['valuation_label'] == 'Undervalued'].sort_values('value_gap_pct').head(20)[['name', 'team', 'current_value', 'predicted_value', 'value_gap_pct']])

with t2:
    st.table(df[df['valuation_label'] == 'Overvalued'].sort_values('value_gap_pct', ascending=False).head(20)[['name', 'team', 'current_value', 'predicted_value', 'value_gap_pct']])
