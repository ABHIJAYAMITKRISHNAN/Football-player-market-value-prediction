import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from scipy import stats
from frontend.components.styles import apply_custom_styles
from src.preprocessing import load_and_preprocess

st.set_page_config(page_title="Player Archetypes", layout="wide")
apply_custom_styles()

PLOTLY_THEME = {
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'font': {'color': '#e6edf3', 'family': 'Inter, sans-serif'},
    'xaxis': {'gridcolor': '#21262d', 'linecolor': '#30363d'},
    'yaxis': {'gridcolor': '#21262d', 'linecolor': '#30363d'},
    'margin': {'l': 40, 'r': 40, 't': 50, 'b': 40},
    'hoverlabel': {'bgcolor': '#161b22', 'bordercolor': '#30363d', 'font_color': '#e6edf3'},
    'legend': {'bgcolor': 'rgba(0,0,0,0)', 'bordercolor': '#30363d'}
}

CLUSTER_COLORS = ['#00ff87', '#ffd700', '#00d4ff', '#c084fc', '#fb923c', '#f85149', '#38bdf8']

ARCHETYPE_DESCRIPTIONS = {
    "Elite Prospects": "Highly touted young talents with immense potential and rising market value.",
    "Defensive Stalwarts": "Reliable defensive anchors focused on clean sheets and stability.",
    "Creative Engine": "Playmakers who consistently generate assists and control the tempo.",
    "Injury-Prone Veterans": "Older players with a history of injuries impacting their consistency.",
    "Versatile Squad Players": "Flexible utility athletes providing depth across multiple positions.",
    "High-Output Attackers": "Premium forwards characterized by exceptional goal-scoring frequency.",
    "Emerging Talents": "Breakthrough young players beginning to make their mark on the first team."
}

CLUSTER_LABELS = {
    0: "Elite Prospects",
    1: "Defensive Stalwarts",
    2: "Creative Engine",
    3: "Injury-Prone Veterans",
    4: "Versatile Squad Players",
    5: "High-Output Attackers",
    6: "Emerging Talents"
}

@st.cache_data
def get_clustered_data():
    df = load_and_preprocess('data/football.csv')
    kmeans = joblib.load('saved_models/kmeans_clusters.pkl')
    
    from sklearn.preprocessing import StandardScaler
    from src.utils import CLUSTER_FEATURES
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[CLUSTER_FEATURES])
    df['cluster'] = kmeans.predict(X_scaled)
    df['archetype'] = df['cluster'].map(CLUSTER_LABELS)
    return df

@st.cache_data
def compute_archetype_centroids(df):
    features = ['goals', 'assists', 'age', 'injury_burden', 'appearance', 'award']
    centroids = df.groupby('archetype')[features].mean()
    
    # Normalize features to [0, 1]
    normed = (centroids - centroids.min()) / (centroids.max() - centroids.min() + 1e-9)
    return normed

df = get_clustered_data()
centroids_df = compute_archetype_centroids(df)

# ==========================================
# Section 1: Radar Chart (full width)
# ==========================================
st.markdown('<div class="section-header">Archetype DNA — What Defines Each Profile</div>', unsafe_allow_html=True)

fig_radar = go.Figure()
features_list = centroids_df.columns.tolist()

for idx, (arch_name, row) in enumerate(centroids_df.iterrows()):
    fig_radar.add_trace(go.Scatterpolar(
        r=row.values.tolist() + [row.values[0]],
        theta=features_list + [features_list[0]],
        fill='toself',
        opacity=0.5,
        name=arch_name,
        line=dict(color=CLUSTER_COLORS[idx % len(CLUSTER_COLORS)])
    ))

fig_radar.update_layout(
    polar=dict(
        radialaxis=dict(visible=True, range=[0, 1], gridcolor='#30363d', linecolor='#30363d'),
        angularaxis=dict(gridcolor='#30363d', linecolor='#30363d')
    ),
    **PLOTLY_THEME
)

st.plotly_chart(fig_radar, use_container_width=True)

# ==========================================
# Section 2: Bubble Chart (full width)
# ==========================================
st.markdown('<div class="section-header">Player Distribution — Age vs Output vs Value</div>', unsafe_allow_html=True)

# Clip productivity_score to 95th percentile
p95 = df['productivity_score'].quantile(0.95)
df_bubble = df.copy()
df_bubble['clipped_output'] = df_bubble['productivity_score'].clip(upper=p95)

# Calculate bubble sizes
val_min = df_bubble['current_value'].min()
val_max = df_bubble['current_value'].max()
df_bubble['bubble_size'] = 4 + 32 * ((df_bubble['current_value'] - val_min) / (val_max - val_min + 1e-9))

# Format current value for hover
df_bubble['formatted_value'] = df_bubble['current_value'].apply(lambda x: f"€{x:,.0f}")

fig_bubble = px.scatter(
    df_bubble,
    x='clipped_output',
    y='age',
    size='bubble_size',
    color='archetype',
    hover_name='name',
    hover_data={'team': True, 'formatted_value': True, 'bubble_size': False, 'archetype': False, 'clipped_output': False, 'age': False},
    color_discrete_sequence=CLUSTER_COLORS
)

# Apply size manually for px.scatter to allow max marker size interpretation properly
fig_bubble.update_traces(marker=dict(sizeref=1, sizemode='diameter', opacity=0.7))

fig_bubble.update_layout(**PLOTLY_THEME)
fig_bubble.update_xaxes(title_text="Productivity Score (Clipped)")
fig_bubble.update_yaxes(range=[15, 42], title_text="Age")

st.plotly_chart(fig_bubble, use_container_width=True)

# ==========================================
# Section 3: Player Search
# ==========================================
st.markdown('<div class="section-header">Search a Player</div>', unsafe_allow_html=True)

search_list = sorted(df['name'].tolist())
selected_player = st.selectbox("Find Player Profile", options=[""] + search_list)

if selected_player:
    player_data = df[df['name'] == selected_player].iloc[0]
    arch_name = player_data['archetype']
    
    # Calculate percentile within archetype
    arch_subset = df[df['archetype'] == arch_name]
    percentile = stats.percentileofscore(arch_subset['productivity_score'], player_data['productivity_score'])
    
    color_idx = list(CLUSTER_LABELS.values()).index(arch_name) % len(CLUSTER_COLORS)
    badge_color = CLUSTER_COLORS[color_idx]
    
    description = ARCHETYPE_DESCRIPTIONS.get(arch_name, "A distinct player profile.")
    
    st.markdown(f"""
        <div style="background-color: #161b22; padding: 24px; border-radius: 8px; border: 1px solid #30363d; margin-top: 10px;">
            <h2 style="color: #00ff87; margin-top: 0;">{player_data['name']}</h2>
            <div style="margin-bottom: 12px;">
                <span style="background-color: rgba(255,255,255,0.1); color: {badge_color}; border: 1px solid {badge_color}; padding: 4px 12px; border-radius: 12px; font-weight: bold;">
                    {arch_name}
                </span>
            </div>
            <p style="color: #e6edf3; font-size: 1.1em; margin-bottom: 8px;">{description}</p>
            <p style="color: #8b949e; margin-bottom: 4px;">Top <b>{100 - int(percentile)}%</b> in their archetype based on productivity.</p>
            <p style="color: #8b949e; margin-bottom: 0;">Current Value: <span style="color: #e6edf3; font-weight: bold;">€{player_data['current_value']:,.0f}</span></p>
        </div>
    """, unsafe_allow_html=True)

# ==========================================
# Section 4: Archetype Details Expander
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("View All Archetype Profiles"):
    cols = st.columns(7)
    
    arch_counts = df['archetype'].value_counts()
    arch_val_avg = df.groupby('archetype')['current_value'].mean()
    
    for i, arch_name in enumerate(CLUSTER_LABELS.values()):
        with cols[i]:
            count = arch_counts.get(arch_name, 0)
            avg_val = arch_val_avg.get(arch_name, 0)
            color = CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
            
            st.markdown(f"""
                <div style="text-align: center; border: 1px solid #30363d; border-radius: 6px; padding: 12px; height: 100%; background-color: #161b22;">
                    <div style="width: 24px; height: 24px; background-color: {color}; border-radius: 50%; margin: 0 auto 8px;"></div>
                    <div style="font-weight: bold; font-size: 0.9em; margin-bottom: 6px; color: #e6edf3;">{arch_name}</div>
                    <div style="font-size: 0.8em; color: #8b949e;">{count} players</div>
                    <div style="font-size: 0.8em; color: #8b949e;">avg €{avg_val/1e6:.1f}M</div>
                </div>
            """, unsafe_allow_html=True)
