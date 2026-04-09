import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt
from src.utils import CLUSTER_FEATURES

def train_segmentation_model(df):
    X = df[CLUSTER_FEATURES]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal k - for this project we'll use a fixed good k or automated if possible
    # README suggested k=6-9. We'll use k=7 as a default peak for simplicity in this script.
    k = 7
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_transform(X_scaled).argmin(axis=1) # fit_predict equivalent
    df['cluster'] = kmeans.labels_
    
    # PCA for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['pca1'] = X_pca[:, 0]
    df['pca2'] = X_pca[:, 1]
    
    # Save model
    joblib.dump(kmeans, 'saved_models/kmeans_clusters.pkl')
    
    # Centroid labeling (conceptual, will be used in frontend)
    centroids = kmeans.cluster_centers_
    
    return kmeans, df
