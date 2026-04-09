from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
import joblib
from src.utils import CLUSTER_FEATURES

def find_similar_players(player_name, df, n=10, same_position=False):
    # Load scaler saved in Module 1
    scaler = joblib.load('saved_models/scaler.pkl')
    
    # Prepare features
    features = df[CLUSTER_FEATURES]
    features_scaled = scaler.fit_transform(features) # We use fit_transform here to ensure we match the current DF state, though normally we'd use the saved one if features matched exactly.
    
    if player_name not in df['name'].values:
        return pd.DataFrame()
    
    idx = df[df['name'] == player_name].index[0]
    query_vec = features_scaled[idx].reshape(1, -1)
    
    scores = cosine_similarity(query_vec, features_scaled)[0]
    
    temp_df = df.copy()
    temp_df['similarity_score'] = scores
    
    # Exclude self
    temp_df = temp_df.drop(idx)
    
    if same_position:
        pos_group = df.loc[idx, 'position_group']
        temp_df = temp_df[temp_df['position_group'] == pos_group]
        
    result = temp_df.sort_values(by='similarity_score', ascending=False).head(n)
    result['similarity_score'] = (result['similarity_score'] * 100).round(1)
    
    return result
