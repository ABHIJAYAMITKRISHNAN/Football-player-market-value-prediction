import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from frontend.components.styles import apply_custom_styles
from frontend.components.charts import apply_theme
from src.preprocessing import load_and_preprocess

st.set_page_config(page_title="Player Archetypes", layout="wide")
apply_custom_styles()

@st.cache_data
def get_clustered_data():
    df = load_and_preprocess('data/football.csv')
    kmeans = joblib.load('saved_models/kmeans_clusters.pkl')
    # Use PCA precomputed in training or recompute here for clustering visualization
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from src.utils import CLUSTER_FEATURES
    
    X = df[CLUSTER_FEATURES]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    df['cluster'] = kmeans.predict(X_scaled)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['pca1'] = X_pca[:, 0]
    df['pca2'] = X_pca[:, 1]
    
    return df

df = get_clustered_data()

st.markdown('<div class="section-header">👥 Player Archetype Segmentation</div>', unsafe_allow_html=True)

# Define human labels for clusters (conceptual labels)
cluster_labels = {
    0: "Elite Prospects",
    1: "Defensive Stalwarts",
    2: "Creative Engine",
    3: "Injury-Prone Veterans",
    4: "Versatile Squad Players",
    5: "High-Output Attackers",
    6: "Emerging Talents"
}
df['archetype'] = df['cluster'].map(cluster_labels)

fig = px.scatter(df, x='pca1', y='pca2', color='archetype', hover_name='name',
                 hover_data=['team', 'position', 'current_value'],
                 color_discrete_sequence=px.colors.qualitative.Pastel)

st.plotly_chart(apply_theme(fig), use_container_width=True)

search = st.text_input("Highlight Player:")
if search:
    match = df[df['name'].str.contains(search, case=False)]
    if not match.empty:
        st.write(f"**{match.iloc[0]['name']}** belongs to the archetype: **{match.iloc[0]['archetype']}**")
