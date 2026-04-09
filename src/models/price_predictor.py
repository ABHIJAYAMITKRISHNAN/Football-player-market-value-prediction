import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap
import joblib
import os
import matplotlib.pyplot as plt
from src.utils import BASE_FEATURES, TARGET_VALUE

def train_price_model(df):
    X = df[BASE_FEATURES]
    y = np.log1p(df[TARGET_VALUE])
    
    # Train/Test Split stratified by position group
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df['position_group_encoded']
    )
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for reuse
    joblib.dump(scaler, 'saved_models/scaler.pkl')
    
    # XGBoost with tuning
    param_dist = {
        'n_estimators': [200, 300, 500],
        'max_depth': [4, 6, 8],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    random_search = RandomizedSearchCV(xgb_model, param_dist, n_iter=10, cv=5, scoring='r2', n_jobs=-1, random_state=42)
    random_search.fit(X_train_scaled, y_train)
    
    best_model = random_search.best_estimator_
    
    # Evaluation
    preds = best_model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"Price Predictor Metrics: RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    
    # SHAP
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test_scaled)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=BASE_FEATURES, show=False)
    plt.savefig('saved_models/shap_price_summary.png', bbox_inches='tight')
    plt.close()
    
    # Save model
    joblib.dump(best_model, 'saved_models/price_predictor.pkl')
    
    return best_model
