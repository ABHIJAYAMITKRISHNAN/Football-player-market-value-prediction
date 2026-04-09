import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import shap
import matplotlib.pyplot as plt
from src.utils import INJURY_FEATURES, TARGET_INJURY

def train_injury_model(df):
    X = df[INJURY_FEATURES]
    y = df[TARGET_INJURY]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluation
    preds = model.predict(X_test)
    print("Injury risk classifier prediction report:")
    print(classification_report(y_test, preds))
    
    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # shap_values is a list for multiclass. We'll plot summary for all.
    # To avoid UI blocking, we save it.
    # Note: SHAP summary plot for multiclass is slightly different.
    
    joblib.dump(model, 'saved_models/injury_classifier.pkl')
    
    return model
