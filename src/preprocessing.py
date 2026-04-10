import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def load_and_preprocess(path="data/football.csv"):
    df = pd.read_csv(path)
    
    # Drop duplicates
    df = df.drop_duplicates(subset=['player'])
    
    # Missing value handling
    num_cols = [
        'height', 'appearance', 'goals', 'assists', 'yellow cards', 
        'second yellow cards', 'red cards', 'goals conceded', 'clean sheets', 
        'minutes played', 'days_injured', 'games_injured', 'award', 
        'current_value', 'highest_value'
    ]
    df[num_cols] = df[num_cols].fillna(0)
    
    cat_cols = ['position', 'team']
    df[cat_cols] = df[cat_cols].fillna('Unknown')
    
    # Drop records where both current and highest value are 0
    df = df[~((df['current_value'] == 0) & (df['highest_value'] == 0))]
    
    # Type enforcement
    int_cols = [
        'height', 'age', 'appearance', 'second yellow cards', 'red cards',
        'minutes played', 'days_injured', 'games_injured', 'award',
        'current_value', 'highest_value', 'position_encoded', 'winger'
    ]
    float_cols = ['goals', 'assists', 'yellow cards', 'goals conceded', 'clean sheets']
    
    df[int_cols] = df[int_cols].fillna(0).astype(int)
    df[float_cols] = df[float_cols].fillna(0.0).astype(float)
    
    # Outlier clipping removed to prevent capping top players' values at 45M
    
    # Feature engineering
    df['value_drop_ratio'] = (df['highest_value'] - df['current_value']) / (df['highest_value'] + 1)
    df['injury_burden'] = df['days_injured'] / (df['appearance'] + 1)
    df['productivity_score'] = df['goals'] + df['assists']
    
    # Per-90 metrics (handle division by zero)
    safe_minutes = df['minutes played'].replace(0, np.nan)
    df['goals_per_90'] = (df['goals'] / safe_minutes * 90).fillna(0)
    df['assists_per_90'] = (df['assists'] / safe_minutes * 90).fillna(0)
    df['productivity_per_90'] = df['goals_per_90'] + df['assists_per_90']
    
    # Target Encoding for team (replace team label with avg market value of that team)
    team_means = df.groupby('team')['current_value'].mean()
    df['team_target_encoded'] = df['team'].map(team_means)
    # Save the mapping for inference/frontend usage
    os.makedirs('saved_models', exist_ok=True)
    joblib.dump(team_means.to_dict(), 'saved_models/team_target_encoding.pkl')
    
    # Age groups
    def age_bin(age):
        if age <= 21: return 'Youth (<=21)'
        if age <= 28: return 'Prime (22-28)'
        if age <= 32: return 'Senior (29-32)'
        return 'Veteran (33+)'
    df['age_group'] = df['age'].apply(age_bin)
    
    # Position groups
    def pos_group(pos):
        pos_lower = str(pos).lower()
        if 'goalkeeper' in pos_lower: return 'GK'
        if 'defender' in pos_lower: return 'DEF'
        if 'midfield' in pos_lower: return 'MID'
        if 'attack' in pos_lower: return 'FWD'
        return 'MID' # Default
    df['position_group'] = df['position'].apply(pos_group)
    
    # Injury risk label
    def label_injury_risk(row):
        score = 0
        if row['days_injured'] > 180: score += 2
        elif row['days_injured'] > 60: score += 1
        if row['games_injured'] > 20: score += 2
        elif row['games_injured'] > 7: score += 1
        if row['age'] > 30: score += 1
        if score >= 4: return 'High'
        elif score >= 2: return 'Medium'
        else: return 'Low'
    df['injury_risk_label'] = df.apply(label_injury_risk, axis=1)
    
    # Encoding (keep label encoding for remaining categorical columns)
    encode_cols = ['position', 'team', 'position_group', 'age_group', 'injury_risk_label']
    label_encoders = {}
    
    for col in encode_cols:
        le = LabelEncoder()
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
        
    joblib.dump(label_encoders, 'saved_models/label_encoders.pkl')
    
    return df
