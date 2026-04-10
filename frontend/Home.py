import streamlit as st
import pandas as pd
import plotly.express as px
from streamlit_option_menu import option_menu
from frontend.components.styles import apply_custom_styles
from frontend.components.charts import apply_theme
import os

st.set_page_config(page_title="Football Transfer Intelligence", layout="wide", initial_sidebar_state="expanded")

# Inject CSS
apply_custom_styles()

@st.cache_data
def get_data():
    if not os.path.exists('data/football.csv'):
        return pd.DataFrame()
    
    df = pd.read_csv('data/football.csv')
    
    # Engineer position_group inline
    df['position_group'] = 'MID' # Default
    pos_lower = df['position'].astype(str).str.lower()
    
    df.loc[pos_lower.str.contains('goalkeeper', na=False), 'position_group'] = 'GK'
    df.loc[pos_lower.str.contains('defender', na=False), 'position_group'] = 'DEF'
    df.loc[pos_lower.str.contains('midfield', na=False), 'position_group'] = 'MID'
    df.loc[pos_lower.str.contains('attack', na=False), 'position_group'] = 'FWD'
    
    return df

df = get_data()

if df.empty:
    st.error("Dataset not found. Please ensure data/football.csv exists.")
    st.stop()


# Main Dashboard
st.markdown('<div class="section-header">🌍 Global Market Overview</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Players", f"{len(df):,}")
with col2:
    avg_val = df['current_value'].mean()
    st.metric("Avg Market Value", f"€{avg_val/1e6:.1f}M")
with col3:
    top_player = df.loc[df['current_value'].idxmax()]
    st.metric("Most Valuable", top_player['name'], delta=f"€{top_player['current_value']/1e6:.0f}M")
with col4:
    # Most injury prone position group
    injury_prone = df.groupby('position_group')['days_injured'].mean().idxmax()
    st.metric("Top Position", injury_prone)

# Addition 1: Top 5 Most Valuable Players Strip
st.markdown("#### 🏅 Most Valuable Players")
top5 = df.nlargest(5, 'current_value')
cols5 = st.columns(5)
pos_colors = {'GK': '#38bdf8', 'DEF': '#00ff87', 'MID': '#c084fc', 'FWD': '#ffd700'}

for i, (_, row) in enumerate(top5.iterrows()):
    with cols5[i]:
        badge_color = pos_colors.get(row['position_group'], '#8b949e')
        st.markdown(f"""
        <div style="background:#161b22; border:1px solid #30363d; border-radius:10px; padding:14px; text-align:center;">
            <div style="font-weight:bold; color:#e6edf3; font-size:1.1rem; margin-bottom:8px;">{row['name']}</div>
            <div style="margin-bottom:8px;"><span style="background-color:rgba(255,255,255,0.1); color:{badge_color}; border:1px solid {badge_color}; padding:2px 8px; border-radius:12px; font-size:0.8rem; font-weight:bold;">{row['position_group']}</span></div>
            <div style="color:#00ff87; font-weight:bold; font-size:1.2rem;">€{row['current_value']/1e6:.0f}M</div>
        </div>
        """, unsafe_allow_html=True)

# Addition 2: Market Snapshot Metrics Row
st.divider()
st.markdown("#### 📊 Market Snapshot")
snap_cols = st.columns(3)

with snap_cols[0]:
    youth_avg = df[df['age'] <= 23]['current_value'].mean()
    vet_avg = df[df['age'] >= 30]['current_value'].mean()
    youth_diff = youth_avg - vet_avg
    st.metric("Youth Premium (≤ 23 vs ≥ 30)", f"€{youth_avg/1e6:.1f}M", delta=f"€{youth_diff/1e6:+.1f}M")

with snap_cols[1]:
    # Engineer condition inline: days_injured > 180 OR games_injured > 20
    high_risk_mask = (df['days_injured'] > 180) | (df['games_injured'] > 20)
    high_risk_pct = (high_risk_mask.sum() / len(df)) * 100
    st.metric("High Injury Risk %", f"{high_risk_pct:.1f}%", delta=f"of {len(df)} players", delta_color="off")

with snap_cols[2]:
    top_club = df['team'].value_counts().idxmax()
    top_club_count = df['team'].value_counts().max()
    st.metric("Most Represented Club", top_club, delta=f"{top_club_count} players", delta_color="off")

# Existing Charts
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

# Addition 3: Module Navigation Cards Grid
st.divider()
st.markdown("#### 🧭 Explore Modules")

nav_data = [
    {"emoji": "💰", "name": "Price Prediction", "desc": "Estimate a player's market value using ML", "path": "pages/01_Price_Prediction.py"},
    {"emoji": "📉", "name": "Depreciation Forecast", "desc": "Predict if a player's value will rise or fall", "path": "pages/02_Depreciation.py"},
    {"emoji": "👥", "name": "Player Archetypes", "desc": "Discover which profile a player belongs to", "path": "pages/03_Archetypes.py"},
    {"emoji": "🔍", "name": "Similar Players", "desc": "Find the closest statistical alternatives", "path": "pages/04_Similar_Players.py"},
    {"emoji": "⚖️", "name": "Valuation Detector", "desc": "Spot overvalued and undervalued players", "path": "pages/05_Valuation.py"},
    {"emoji": "🏆", "name": "Performance Ranking", "desc": "See who ranks best in each position", "path": "pages/06_Performance_Ranking.py"},
    {"emoji": "🏥", "name": "Injury Risk", "desc": "Assess a player's injury risk level", "path": "pages/07_Injury_Risk.py"}
]

# Row 1
nav_cols1 = st.columns(4)
for i in range(4):
    with nav_cols1[i]:
        item = nav_data[i]
        st.markdown(f"""
        <div style="background:#161b22; border:1px solid #30363d; border-radius:12px; padding:18px; text-align:center; min-height:160px; margin-bottom:10px;">
            <div style="font-size:2rem; margin-bottom:8px;">{item['emoji']}</div>
            <div style="font-weight:bold; color:#e6edf3; font-size:1.1rem; margin-bottom:4px;">{item['name']}</div>
            <div style="color:#8b949e; font-size:0.85rem;">{item['desc']}</div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link(item['path'], label=item['name'] + " →")

# Row 2
nav_cols2 = st.columns(3)
for i in range(4, 7):
    with nav_cols2[i-4]:
        item = nav_data[i]
        st.markdown(f"""
        <div style="background:#161b22; border:1px solid #30363d; border-radius:12px; padding:18px; text-align:center; min-height:160px; margin-bottom:10px;">
            <div style="font-size:2rem; margin-bottom:8px;">{item['emoji']}</div>
            <div style="font-weight:bold; color:#e6edf3; font-size:1.1rem; margin-bottom:4px;">{item['name']}</div>
            <div style="color:#8b949e; font-size:0.85rem;">{item['desc']}</div>
        </div>
        """, unsafe_allow_html=True)
        st.page_link(item['path'], label=item['name'] + " →")
