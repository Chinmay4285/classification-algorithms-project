"""
Logistic Regression Implementation for Titanic Survival Prediction
Optimized for NVIDIA RTX 3060 GPU (6GB VRAM)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import time

from utils.data_utils import load_titanic_data, get_gpu_memory_info
from utils.evaluation import ModelEvaluator, measure_prediction_time
from utils.visualization import (plot_data_distribution, plot_correlation_matrix, 
                               plot_feature_importance, plot_pca_visualization)

def train_logistic_regression():
    """Train and evaluate Logistic Regression model."""
    
    print("[START] Logistic Regression - Titanic Survival Prediction")
    print("=" * 60)
    
    # Check GPU info
    gpu_info = get_gpu_memory_info()
    print(f"[SAVE] GPU Info: {gpu_info}")
    print()
    
    # Load and prepare data
    print("[CHART] Loading Titanic dataset...")
    X_train, X_test, y_train, y_test, feature_names = load_titanic_data()
    
    print(f"[INFO] Dataset Info:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Feature names: {feature_names}")
    print()
    
    # Create visualizations
    print("[CHART] Creating data visualizations...")
    
    # Data distribution
    fig1 = plot_data_distribution(X_train, y_train, feature_names, 
                                 title="Titanic Dataset - Feature Distributions")
    plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Correlation matrix
    fig2 = plot_correlation_matrix(X_train, feature_names)
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # PCA visualization
    fig3 = plot_pca_visualization(X_train, y_train, feature_names)
    plt.savefig('pca_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("[OK] Visualizations saved!")
    print()
    
    # Train basic model
    print("[TRAIN] Training Logistic Regression model...")
    start_time = time.time()
    
    # Initialize model with basic parameters
    model = LogisticRegression(
        C=1.0,                    # Regularization strength
        penalty='l2',             # L2 regularization
        solver='liblinear',       # Good for small datasets
        random_state=42,
        max_iter=1000
    )
    
    # Fit the model
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"[OK] Training completed in {training_time:.2f} seconds")
    print()
    
    # Make predictions
    print("[PREDICT] Making predictions...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Measure prediction time
    pred_time = measure_prediction_time(model, X_test)
    print(f"[FAST] Average prediction time: {pred_time:.4f} seconds")
    print()
    
    # Evaluate model
    print("[CHART] Evaluating model performance...")
    evaluator = ModelEvaluator("Logistic Regression")
    evaluator.results['training_time'] = training_time
    evaluator.results['prediction_time'] = pred_time
    
    metrics = evaluator.evaluate_classification(
        y_test, y_pred, y_proba, 
        class_names=['Died', 'Survived']
    )
    
    evaluator.print_detailed_report()
    
    # Cross-validation
    print("\n[PROCESS] Performing 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance (coefficients)
    print("\n[TARGET] Analyzing feature importance...")
    coefficients = model.coef_[0]
    
    # Plot feature importance
    fig4 = plot_feature_importance(
        np.abs(coefficients), feature_names,
        title="Logistic Regression - Feature Importance (|Coefficients|)"
    )
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print coefficients with interpretation
    print("\n[INFO] Feature Coefficients (with interpretation):")
    for name, coef in zip(feature_names, coefficients):
        effect = "increases" if coef > 0 else "decreases"
        print(f"  {name:12}: {coef:+7.3f} ({effect} survival probability)")
    print()
    
    # Hyperparameter tuning
    print("[PARAM] Hyperparameter tuning with GridSearchCV...")
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0, 100.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }
    
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    print(f"[BEST] Best parameters: {grid_search.best_params_}")
    print(f"[TARGET] Best CV score: {grid_search.best_score_:.4f}")
    print()
    
    # Evaluate best model
    print("[START] Evaluating optimized model...")
    y_pred_best = best_model.predict(X_test)
    y_proba_best = best_model.predict_proba(X_test)
    
    evaluator_best = ModelEvaluator("Logistic Regression (Optimized)")
    metrics_best = evaluator_best.evaluate_classification(
        y_test, y_pred_best, y_proba_best,
        class_names=['Died', 'Survived']
    )
    
    print(f"\n[CHART] Performance Comparison:")
    print(f"  Basic Model Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  Optimized Model Accuracy: {metrics_best['accuracy']:.4f}")
    print(f"  Improvement:              {metrics_best['accuracy'] - metrics['accuracy']:+.4f}")
    print()
    
    # Create evaluation plots
    print("[CHART] Creating evaluation plots...")
    
    # Confusion matrix
    fig5 = evaluator_best.plot_confusion_matrix(
        y_test, y_pred_best, 
        class_names=['Died', 'Survived']
    )
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC curve
    fig6 = evaluator_best.plot_roc_curve(y_test, y_proba_best)
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Precision-Recall curve
    fig7 = evaluator_best.plot_precision_recall_curve(y_test, y_proba_best)
    plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("[OK] Evaluation plots saved!")
    print()
    
    # Sample predictions with explanations
    print("[SEARCH] Sample Predictions with Explanations:")
    print("-" * 60)
    
    # Get some sample predictions
    sample_indices = [0, 1, 2, 10, 20]
    X_sample = X_test[sample_indices]
    y_sample_true = y_test[sample_indices]
    y_sample_pred = best_model.predict(X_sample)
    y_sample_proba = best_model.predict_proba(X_sample)
    
    for i, idx in enumerate(sample_indices):
        prob_died = y_sample_proba[i][0]
        prob_survived = y_sample_proba[i][1]
        prediction = "Survived" if y_sample_pred[i] == 1 else "Died"
        actual = "Survived" if y_sample_true[i] == 1 else "Died"
        correct = "[OK]" if y_sample_pred[i] == y_sample_true[i] else "[ERROR]"
        
        print(f"Sample {idx+1}:")
        print(f"  Prediction: {prediction} (Confidence: {max(prob_died, prob_survived):.1%}) {correct}")
        print(f"  Actual: {actual}")
        print(f"  Probabilities: Died={prob_died:.3f}, Survived={prob_survived:.3f}")
        
        # Feature contributions (simplified interpretation)
        features = X_sample[i]
        coefficients = best_model.coef_[0]
        contributions = features * coefficients
        top_contrib = np.argsort(np.abs(contributions))[-3:][::-1]
        
        print(f"  Top Contributing Features:")
        for feat_idx in top_contrib:
            contrib = contributions[feat_idx]
            direction = "toward survival" if contrib > 0 else "toward death"
            print(f"    {feature_names[feat_idx]}: {contrib:+.3f} ({direction})")
        print()
    
    # Save the trained model
    print("[SAVE] Saving trained model...")
    joblib.dump(best_model, 'logistic_regression_model.pkl')
    joblib.dump(feature_names, 'feature_names.pkl')
    
    # Save results summary
    results_summary = {
        'model_name': 'Logistic Regression',
        'dataset': 'Titanic Survival',
        'training_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'features': X_train.shape[1],
        'training_time': training_time,
        'prediction_time': pred_time,
        'best_params': grid_search.best_params_,
        'basic_accuracy': metrics['accuracy'],
        'optimized_accuracy': metrics_best['accuracy'],
        'cross_val_score': cv_scores.mean(),
        'cross_val_std': cv_scores.std()
    }
    
    pd.Series(results_summary).to_json('results_summary.json', indent=2)
    
    print("[OK] Model and results saved!")
    print()
    print("[TARGET] Summary:")
    print(f"  Final Accuracy: {metrics_best['accuracy']:.1%}")
    print(f"  Training Time: {training_time:.2f}s")
    print(f"  Prediction Speed: {1/pred_time:.0f} predictions/second")
    print()
    print("[FOLDER] Generated Files:")
    print("  - logistic_regression_model.pkl (trained model)")
    print("  - feature_names.pkl (feature information)")
    print("  - results_summary.json (performance metrics)")
    print("  - *.png (visualization plots)")
    print()
    print("[START] Next: Run 'jupyter notebook explanation.ipynb' for detailed analysis!")

if __name__ == "__main__":
    train_logistic_regression()