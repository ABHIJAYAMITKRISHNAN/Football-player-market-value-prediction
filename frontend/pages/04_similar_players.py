import streamlit as st
import pandas as pd
from frontend.components.styles import apply_custom_styles
from frontend.components.charts import create_radar_chart
from src.preprocessing import load_and_preprocess
from src.models.similarity import find_similar_players

st.set_page_config(page_title="Similar Players", layout="wide")
apply_custom_styles()

@st.cache_data
def get_data():
    return load_and_preprocess('data/football.csv')

df = get_data()

st.markdown('<div class="section-header">🔍 Similar Player Discovery</div>', unsafe_allow_html=True)

player_name = st.selectbox("Find players similar to:", options=[""] + sorted(df['name'].tolist()))
same_pos = st.checkbox("Same Position Group Only")

if player_name:
    results = find_similar_players(player_name, df, same_position=same_pos)
    
    if not results.empty:
        st.subheader(f"Top 10 Matches for {player_name}")
        
        # Display as table
        display_df = results[['name', 'team', 'position', 'age', 'current_value', 'similarity_score']].copy()
        display_df['current_value'] = display_df['current_value'].apply(lambda x: f"€{x:,.0f}")
        st.table(display_df)
        
        st.markdown("---")
        # Radar Comparison with #1 Match
        best_match = results.iloc[0]
        st.subheader(f"Comparison: {player_name} vs {best_match['name']}")
        
        radar_cols = ['goals', 'assists', 'appearance', 'minutes played', 'award', 'injury_burden']
        # Normalize for radar locally
        q_row = df[df['name'] == player_name].iloc[0]
        m_row = best_match
        
        # Simple normalize
        q_vals = []
        m_vals = []
        for col in radar_cols:
            max_v = df[col].max() + 1e-9
            q_vals.append(q_row[col] / max_v)
            m_vals.append(m_row[col] / max_v)
            
        fig = create_radar_chart(radar_cols, q_vals, m_vals, player_name, best_match['name'])
        st.plotly_chart(fig, use_container_width=True)
