import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from src.utils import BASE_FEATURES, TARGET_DEPRECIATION

def label_trajectory(row):
    ratio = row['value_drop_ratio']
    if ratio < 0.10:
        return 0 # 'Appreciating'
    elif ratio < 0.40:
        return 1 # 'Stable'
    else:
        return 2 # 'Depreciating'

def train_depreciation_model(df):
    df['trajectory_label'] = df.apply(label_trajectory, axis=1)
    
    # Features specific to depreciation
    dep_features = [
        'age', 'age_group_encoded', 'appearance', 'injury_burden', 
        'days_injured', 'productivity_score', 'award', 'position_encoded'
    ]
    
    X = df[dep_features]
    y = df['trajectory_label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # XGBoost Classifier
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        random_state=42,
        class_weight='balanced' # Note: XGBoost handles weights via 'scale_pos_weight' usually or custom weights, but sklearn-api often supports 'sample_weight' in fit. 
                                # For multiclass, we'll rely on balanced data or manual weighting if needed.
    )
    
    # Manual weighting for imbalanced classes
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    # Evaluation
    preds = model.predict(X_test)
    print("Depreciation Forecaster Report:")
    print(classification_report(y_test, preds, target_names=['Appreciating', 'Stable', 'Depreciating']))
    
    joblib.dump(model, 'saved_models/depreciation_model.pkl')
    
    return model
