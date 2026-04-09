import streamlit as st
import pandas as pd
import plotly.express as px
from frontend.components.styles import apply_custom_styles
from frontend.components.charts import apply_theme
import os

st.set_page_config(page_title="Performance Rankings", layout="wide")
apply_custom_styles()

if not os.path.exists('saved_models/performance_rankings.parquet'):
    st.error("Performance rankings not found. Please run train_all.py first.")
    st.stop()

df = pd.read_parquet('saved_models/performance_rankings.parquet')

st.markdown('<div class="section-header">🏆 Player Performance Leaderboards</div>', unsafe_allow_html=True)

tabs = st.tabs(["FWD", "MID", "DEF", "GK"])

for i, group in enumerate(["FWD", "MID", "DEF", "GK"]):
    with tabs[i]:
        group_df = df[df['position_group'] == group].sort_values('performance_score', ascending=False)
        
        st.subheader(f"Top 20 {group}s")
        fig = px.bar(group_df.head(20), y='name', x='performance_score', orientation='h', 
                     color='performance_score', color_continuous_scale='Viridis')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(apply_theme(fig), use_container_width=True)
        
        st.dataframe(group_df[['rank', 'name', 'team', 'performance_score', 'current_value']].head(50))
