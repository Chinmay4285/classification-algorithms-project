"""
XGBoost/LightGBM/CatBoost Implementation for Classification
Optimized for NVIDIA RTX 3060 GPU (6GB VRAM)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import XGBClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

from utils.data_utils import load_wine_quality_data, get_gpu_memory_info
from utils.evaluation import ModelEvaluator, measure_prediction_time
from utils.visualization import plot_data_distribution, plot_pca_visualization

def train_xgboost_lightgbm_catboost():
    """Train and evaluate XGBoost/LightGBM/CatBoost model."""
    
    print("[START] XGBoost/LightGBM/CatBoost - Classification")
    print("=" * 60)
    
    # Check GPU info
    gpu_info = get_gpu_memory_info()
    print(f"[SAVE] GPU Info: {gpu_info}")
    print()
    
    # Load data
    print("[CHART] Loading dataset...")
    X_train, X_test, y_train, y_test, feature_names = load_wine_quality_data()
    
    print(f"[INFO] Dataset Info:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")
    print()
    
    # Train model
    print("[TRAIN] Training XGBoost/LightGBM/CatBoost model...")
    start_time = time.time()
    
    model = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"[OK] Training completed in {training_time:.2f} seconds")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    # Evaluate
    evaluator = ModelEvaluator("XGBoost/LightGBM/CatBoost")
    evaluator.results['training_time'] = training_time
    
    metrics = evaluator.evaluate_classification(y_test, y_pred, y_proba)
    evaluator.print_detailed_report()
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"\nCV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    print("\n[TARGET] Summary:")
    print(f"  Final Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Training Time: {training_time:.2f}s")

if __name__ == "__main__":
    train_xgboost_lightgbm_catboost()
