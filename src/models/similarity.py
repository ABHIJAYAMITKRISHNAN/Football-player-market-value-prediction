from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from src.utils import CLUSTER_FEATURES

def find_similar_players(player_name, df, n=10, include_different_positions=False):
    # Prepare custom feature set with price included
    sim_features = CLUSTER_FEATURES + ['current_value']
    features = df[sim_features]
    
    # Scale locally to account for the new price feature
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    if player_name not in df['name'].values:
        return pd.DataFrame()
    
    idx = df[df['name'] == player_name].index[0]
    pos_group = df.loc[idx, 'position_group']
    
    # Weight the features: current_value and position_encoded
    # current_value is the last column (5.0 weight)
    features_scaled[:, -1] *= 5.0 
    
    # position_encoded heavily weights the specific exact role
    pos_idx = sim_features.index('position_encoded')
    features_scaled[:, pos_idx] *= 3.0
    
    # Dynamic weights for attackers and midfielders
    if pos_group in ['FWD', 'MID']:
        if 'goals' in sim_features:
            features_scaled[:, sim_features.index('goals')] *= 7.0
        if 'assists' in sim_features:
            features_scaled[:, sim_features.index('assists')] *= 7.0
        if 'productivity_score' in sim_features:
            features_scaled[:, sim_features.index('productivity_score')] *= 7.0
    query_vec = features_scaled[idx].reshape(1, -1)
    
    scores = cosine_similarity(query_vec, features_scaled)[0]
    
    temp_df = df.copy()
    temp_df['similarity_score'] = scores
    
    # Exclude self
    temp_df = temp_df.drop(idx)
    
    if not include_different_positions:
        temp_df = temp_df[temp_df['position_group'] == pos_group]
        
    result = temp_df.sort_values(by='similarity_score', ascending=False).head(n)
    # Cosine similarity can easily drift when heavily weighted, simply rank them visually or normalize
    # Using MinMax scale locally for presentation if needed, but absolute score drop is fine.
    # Clip max to 100% just in case floating point errs
    result['similarity_score'] = np.clip(result['similarity_score'] * 100, 0, 100).round(1)
    
    return result
