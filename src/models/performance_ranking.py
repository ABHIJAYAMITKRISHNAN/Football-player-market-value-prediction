import pandas as pd
import numpy as np

def compute_rankings(df):
    # Normalize features within position group
    cols_to_norm = [
        'goals', 'assists', 'appearance', 'minutes played', 'award', 
        'clean sheets', 'goals conceded', 'red cards', 'yellow cards'
    ]
    
    ranked_dfs = []
    
    for group, group_df in df.groupby('position_group'):
        g_df = group_df.copy()
        
        # Min-Max Normalization
        for col in cols_to_norm:
            min_val = g_df[col].min()
            max_val = g_df[col].max()
            if max_val > min_val:
                g_df[f'{col}_norm'] = (g_df[col] - min_val) / (max_val - min_val)
            else:
                g_df[f'{col}_norm'] = 0
        
        # Scoring Logic
        if group == 'FWD':
            g_df['performance_score'] = (
                g_df['goals_norm'] * 0.40 + 
                g_df['assists_norm'] * 0.25 + 
                g_df['appearance_norm'] * 0.10 + 
                g_df['minutes played_norm'] * 0.10 + 
                g_df['award_norm'] * 0.15
            )
        elif group == 'MID':
            g_df['performance_score'] = (
                g_df['goals_norm'] * 0.20 + 
                g_df['assists_norm'] * 0.30 + 
                g_df['appearance_norm'] * 0.20 + 
                g_df['minutes played_norm'] * 0.15 + 
                g_df['award_norm'] * 0.15
            )
        elif group == 'DEF':
            # Invert cards
            g_df['red_cards_inv'] = 1 - g_df['red cards_norm']
            g_df['yellow_cards_inv'] = 1 - g_df['yellow cards_norm']
            g_df['performance_score'] = (
                g_df['clean sheets_norm'] * 0.30 + 
                g_df['appearance_norm'] * 0.20 + 
                g_df['minutes played_norm'] * 0.15 + 
                g_df['red_cards_inv'] * 0.15 + 
                g_df['yellow_cards_inv'] * 0.05 + 
                g_df['award_norm'] * 0.15
            )
        elif group == 'GK':
            g_df['goals_conceded_inv'] = 1 - g_df['goals conceded_norm']
            g_df['performance_score'] = (
                g_df['clean sheets_norm'] * 0.40 + 
                g_df['goals_conceded_inv'] * 0.25 + 
                g_df['appearance_norm'] * 0.15 + 
                g_df['minutes played_norm'] * 0.10 + 
                g_df['award_norm'] * 0.10
            )
        else:
            g_df['performance_score'] = 0
            
        g_df['rank'] = g_df['performance_score'].rank(ascending=False)
        ranked_dfs.append(g_df)
        
    full_ranked_df = pd.concat(ranked_dfs)
    full_ranked_df.to_parquet('saved_models/performance_rankings.parquet')
    
    return full_ranked_df
