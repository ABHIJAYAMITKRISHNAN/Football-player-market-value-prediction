import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
from frontend.components.styles import apply_custom_styles, render_kpi
from frontend.components.charts import apply_theme
from src.preprocessing import load_and_preprocess
import os

st.set_page_config(page_title="Football Transfer Intelligence", layout="wide", initial_sidebar_state="expanded")

# Inject CSS
apply_custom_styles()

@st.cache_data
def get_data():
    if os.path.exists('data/football.csv'):
        return load_and_preprocess('data/football.csv')
    return pd.DataFrame()

df = get_data()

if df.empty:
    st.error("Dataset not found. Please ensure data/football.csv exists and run train_all.py")
    st.stop()

# Sidebar Navigation (though pages/ will handle multi-page, app.py is the home)
with st.sidebar:
    st.image("https://img.icons8.com/wired/128/00ff87/soccer-ball.png", width=80)
    st.title("Scout AI")
    st.markdown("---")
    
# Main Dashboard
st.markdown('<div class="section-header">🌍 Global Market Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    render_kpi("Total Players", f"{len(df):,}")
with col2:
    avg_val = df['current_value'].mean()
    render_kpi("Avg Market Value", f"€{avg_val/1e6:.1f}M")
with col3:
    top_player = df.loc[df['current_value'].idxmax()]
    render_kpi("Most Valuable", top_player['name'], f"€{top_player['current_value']/1e6:.0f}M")
with col4:
    # Most injury prone position group
    injury_prone = df.groupby('position_group')['days_injured'].mean().idxmax()
    render_kpi("Most Injury Prone", injury_prone)

st.markdown("---")

c1, c2 = st.columns(2)

with c1:
    st.subheader("Player Distribution by Position")
    fig_pos = px.bar(df['position_group'].value_counts().reset_index(), x='position_group', y='count', 
                     color='position_group', color_discrete_sequence=['#00ff87', '#00d4ff', '#ffd700', '#ff6b6b'])
    st.plotly_chart(apply_theme(fig_pos), use_container_width=True)

with c2:
    st.subheader("Market Value Distribution")
    fig_val = px.histogram(df, x='current_value', nbins=50, log_y=True, color_discrete_sequence=['#00ff87'])
    st.plotly_chart(apply_theme(fig_val), use_container_width=True)

st.markdown('<div class="section-header">🔍 Quick Player Search</div>', unsafe_allow_html=True)
player_search = st.selectbox("Search for a player to see their full profile across modules", 
                             options=[""] + sorted(df['name'].tolist()))

if player_search:
    st.info(f"Navigate to sub-pages to see detailed analysis for {player_search}!")
