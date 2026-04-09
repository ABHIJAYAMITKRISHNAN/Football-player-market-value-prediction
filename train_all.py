"""
Master training script — trains and saves all models.
Run: python train_all.py
"""
import os
import pandas as pd

# Create directory if not exists
os.makedirs("saved_models", exist_ok=True)

from src.preprocessing import load_and_preprocess
from src.models.price_predictor import train_price_model
from src.models.depreciation import train_depreciation_model
from src.models.segmentation import train_segmentation_model
from src.models.injury_classifier import train_injury_model
from src.models.valuation_detector import compute_valuation_gaps
from src.models.performance_ranking import compute_rankings

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    df = load_and_preprocess("data/football.csv")

    print("[1/6] Training price predictor...")
    train_price_model(df)

    print("[2/6] Training depreciation forecaster...")
    train_depreciation_model(df)

    print("[3/6] Training segmentation model...")
    train_segmentation_model(df)

    print("[4/6] Training injury risk classifier...")
    train_injury_model(df)

    print("[5/6] Computing valuation gaps...")
    compute_valuation_gaps(df)

    print("[6/6] Computing performance rankings...")
    compute_rankings(df)

    print("\nAll models trained and saved to saved_models/")
    print("Run: streamlit run frontend/app.py")
