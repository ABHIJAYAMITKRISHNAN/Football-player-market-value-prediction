import pandas as pd
import numpy as np
import joblib
from src.utils import BASE_FEATURES

def compute_valuation_gaps(df):
    model = joblib.load('saved_models/price_predictor.pkl')
    scaler = joblib.load('saved_models/scaler.pkl')
    
    X = df[BASE_FEATURES]
    X_scaled = scaler.transform(X)
    
    df['predicted_value'] = np.expm1(model.predict(X_scaled))
    df['value_gap'] = df['current_value'] - df['predicted_value']
    df['value_gap_pct'] = (df['value_gap'] / (df['predicted_value'] + 1)) * 100
    
    def classify_valuation(pct):
        if pct > 30: return 'Overvalued'
        if pct < -30: return 'Undervalued'
        return 'Fairly Valued'
    
    df['valuation_label'] = df['value_gap_pct'].apply(classify_valuation)
    
    # Save results
    df.to_parquet('saved_models/valuation_results.parquet')
    
    return df
