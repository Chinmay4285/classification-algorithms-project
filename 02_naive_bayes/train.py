"""
Naïve Bayes Implementation for Titanic Survival Prediction
Optimized for NVIDIA RTX 3060 GPU (6GB VRAM)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import joblib
import time

from utils.data_utils import load_spam_email_data, get_gpu_memory_info
from utils.evaluation import ModelEvaluator, measure_prediction_time
from utils.visualization import (plot_data_distribution, plot_correlation_matrix, 
                               plot_pca_visualization)

def analyze_feature_distributions(X_train, y_train, feature_names):
    """Analyze feature distributions per class for Naïve Bayes understanding."""
    
    print("[SEARCH] Analyzing Feature Distributions per Class")
    print("=" * 50)
    
    # Create DataFrame for analysis
    df = pd.DataFrame(X_train, columns=feature_names)
    df['Survived'] = y_train
    
    # Calculate statistics per class
    stats_died = df[df['Survived'] == 0].describe()
    stats_survived = df[df['Survived'] == 1].describe()
    
    print("\n[CHART] Feature Statistics per Class:")
    print("\nClass 0 (Died):")
    print(stats_died.round(3))
    print("\nClass 1 (Survived):")
    print(stats_survived.round(3))
    
    # Visualize distributions
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()
    
    for i, feature in enumerate(feature_names):
        if i < len(axes):
            ax = axes[i]
            
            # Plot histograms for each class
            died_data = df[df['Survived'] == 0][feature]
            survived_data = df[df['Survived'] == 1][feature]
            
            ax.hist(died_data, alpha=0.7, label='Died', bins=20, color='red', density=True)
            ax.hist(survived_data, alpha=0.7, label='Survived', bins=20, color='green', density=True)
            
            ax.set_title(f'{feature}')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_distributions_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df

def compare_naive_bayes_variants(X_train, X_test, y_train, y_test):
    """Compare different Naïve Bayes variants."""
    
    print("[TEST] Comparing Naïve Bayes Variants")
    print("=" * 40)
    
    # Prepare data for different variants
    # Scale data for Multinomial NB (requires non-negative values)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    variants = {
        'Gaussian NB': GaussianNB(),
        'Multinomial NB': MultinomialNB(alpha=1.0),  # Works with non-negative features
        'Bernoulli NB': BernoulliNB(alpha=1.0, binarize=0.5)  # Binarizes features
    }
    
    results = {}
    
    for name, model in variants.items():
        print(f"\n[TRAIN] Training {name}...")
        start_time = time.time()
        
        try:
            if 'Multinomial' in name or 'Bernoulli' in name:
                # Use scaled data for Multinomial and Bernoulli
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)
            else:
                # Use original data for Gaussian
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)
                
            training_time = time.time() - start_time
            
            # Evaluate
            evaluator = ModelEvaluator(name)
            evaluator.results['training_time'] = training_time
            
            metrics = evaluator.evaluate_classification(
                y_test, y_pred, y_proba,
                class_names=['Died', 'Survived']
            )
            
            results[name] = {
                'model': model,
                'metrics': metrics,
                'training_time': training_time,
                'evaluator': evaluator
            }
            
            print(f"   Accuracy: {metrics['accuracy']:.4f}")
            print(f"   Training Time: {training_time:.4f}s")
            
        except Exception as e:
            print(f"   [ERROR] Failed: {e}")
            continue
    
    return results

def train_naive_bayes():
    """Train and evaluate Naïve Bayes models."""
    
    print("[START] Naïve Bayes - Titanic Survival Prediction")
    print("=" * 60)
    
    # Check GPU info
    gpu_info = get_gpu_memory_info()
    print(f"[SAVE] GPU Info: {gpu_info}")
    print()
    
    # Load and prepare data
    print("[CHART] Loading Spam Email dataset (optimal for Naive Bayes)...")
    X_train, X_test, y_train, y_test = load_spam_email_data()
    feature_names = ['email_text']
    
    print(f"[INFO] Dataset Info:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Feature names: {feature_names}")
    print()
    
    # Analyze feature distributions
    df_analysis = analyze_feature_distributions(X_train, y_train, feature_names)
    
    # Create basic visualizations
    print("[CHART] Creating data visualizations...")
    
    # Data distribution
    fig1 = plot_data_distribution(X_train, y_train, feature_names, 
                                 title="Titanic Dataset - Feature Distributions")
    plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # PCA visualization
    fig2 = plot_pca_visualization(X_train, y_train, feature_names)
    plt.savefig('pca_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("[OK] Visualizations saved!")
    print()
    
    # Compare different Naïve Bayes variants
    variant_results = compare_naive_bayes_variants(X_train, X_test, y_train, y_test)
    
    # Select best variant (typically Gaussian for continuous features)
    best_variant = max(variant_results.items(), key=lambda x: x[1]['metrics']['accuracy'])
    best_name = best_variant[0]
    best_result = best_variant[1]
    
    print(f"\n[BEST] Best Variant: {best_name}")
    print(f"   Accuracy: {best_result['metrics']['accuracy']:.4f}")
    
    # Focus on Gaussian Naïve Bayes for detailed analysis
    print("\n[TEST] Detailed Analysis: Gaussian Naïve Bayes")
    print("=" * 45)
    
    # Train Gaussian NB with hyperparameter tuning
    print("[PARAM] Hyperparameter tuning...")
    param_grid = {
        'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    }
    
    grid_search = GridSearchCV(
        GaussianNB(),
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
    
    # Make predictions with best model
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)
    
    # Measure prediction time
    pred_time = measure_prediction_time(best_model, X_test)
    
    # Comprehensive evaluation
    print("\n[CHART] Evaluating optimized model...")
    evaluator = ModelEvaluator("Gaussian Naïve Bayes (Optimized)")
    evaluator.results['training_time'] = best_result['training_time']
    evaluator.results['prediction_time'] = pred_time
    
    metrics = evaluator.evaluate_classification(
        y_test, y_pred, y_proba,
        class_names=['Died', 'Survived']
    )
    
    evaluator.print_detailed_report()
    
    # Cross-validation
    print("\n[PROCESS] Performing 5-fold cross-validation...")
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Analyze class probabilities and feature likelihoods
    print("\n[LEARN] Understanding the Model's Learning")
    print("=" * 40)
    
    # Class priors
    class_priors = np.exp(best_model.class_log_prior_)
    print(f"\n[CHART] Class Priors (learned from data):")
    print(f"   P(Died) = {class_priors[0]:.3f}")
    print(f"   P(Survived) = {class_priors[1]:.3f}")
    
    # Feature statistics learned by the model
    print(f"\n[TARGET] Feature Statistics Learned by Model:")
    print(f"   (Mean and Variance for each class)")
    
    for i, feature_name in enumerate(feature_names):
        mean_died = best_model.theta_[0, i]
        var_died = best_model.var_[0, i]
        mean_survived = best_model.theta_[1, i]
        var_survived = best_model.var_[1, i]
        
        print(f"   {feature_name}:")
        print(f"     Died:     μ={mean_died:.3f}, σ²={var_died:.3f}")
        print(f"     Survived: μ={mean_survived:.3f}, σ²={var_survived:.3f}")
        
        # Interpret difference
        if mean_survived > mean_died:
            print(f"     → Higher values increase survival probability")
        else:
            print(f"     → Lower values increase survival probability")
    
    # Create evaluation plots
    print("\n[CHART] Creating evaluation plots...")
    
    # Confusion matrix
    fig3 = evaluator.plot_confusion_matrix(
        y_test, y_pred, 
        class_names=['Died', 'Survived']
    )
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC curve
    fig4 = evaluator.plot_roc_curve(y_test, y_proba)
    plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Precision-Recall curve
    fig5 = evaluator.plot_precision_recall_curve(y_test, y_proba)
    plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Feature importance visualization (based on class mean differences)
    feature_importance = np.abs(best_model.theta_[1] - best_model.theta_[0])
    
    plt.figure(figsize=(10, 6))
    indices = np.argsort(feature_importance)[::-1]
    plt.barh(range(len(feature_names)), feature_importance[indices],
            color=plt.cm.viridis(np.linspace(0, 1, len(feature_names))))
    plt.yticks(range(len(feature_names)), [feature_names[i] for i in indices])
    plt.xlabel('Mean Difference Between Classes')
    plt.title('Feature Discriminative Power in Naïve Bayes')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_discriminative_power.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("[OK] Evaluation plots saved!")
    
    # Sample predictions with probabilistic reasoning
    print("\n[SEARCH] Sample Predictions with Probabilistic Reasoning:")
    print("-" * 65)
    
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
        print(f"  Probabilities: P(Died)={prob_died:.3f}, P(Survived)={prob_survived:.3f}")
        
        # Show how probabilities are calculated (simplified)
        log_prob_died = best_model.class_log_prior_[0]
        log_prob_survived = best_model.class_log_prior_[1]
        
        print(f"  Calculation breakdown:")
        print(f"    Prior: P(Died)={np.exp(log_prob_died):.3f}, P(Survived)={np.exp(log_prob_survived):.3f}")
        print(f"    Features contribute through Gaussian likelihoods...")
        
        # Show most discriminative features for this sample
        sample_features = X_sample[i]
        feature_contributions = []
        
        for j, (feature_val, feature_name) in enumerate(zip(sample_features, feature_names)):
            mean_diff = best_model.theta_[1, j] - best_model.theta_[0, j]
            contribution = feature_val * mean_diff
            feature_contributions.append((feature_name, contribution))
        
        # Sort by absolute contribution
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"    Top contributing features:")
        for feat_name, contrib in feature_contributions[:3]:
            direction = "survival" if contrib > 0 else "death"
            print(f"      {feat_name}: {contrib:+.3f} (toward {direction})")
        print()
    
    # Compare all variants performance
    print("[BEST] Final Comparison of All Naïve Bayes Variants:")
    print("-" * 55)
    
    for name, result in variant_results.items():
        metrics = result['metrics']
        print(f"{name:20}: Accuracy={metrics['accuracy']:.4f}, "
              f"Time={result['training_time']:.4f}s")
    
    # Save the trained model
    print("\n[SAVE] Saving trained model...")
    joblib.dump(best_model, 'naive_bayes_model.pkl')
    joblib.dump(feature_names, 'feature_names.pkl')
    
    # Save results summary
    results_summary = {
        'model_name': 'Gaussian Naïve Bayes',
        'dataset': 'Titanic Survival',
        'training_samples': X_train.shape[0],
        'test_samples': X_test.shape[0],
        'features': X_train.shape[1],
        'training_time': best_result['training_time'],
        'prediction_time': pred_time,
        'best_params': grid_search.best_params_,
        'accuracy': metrics['accuracy'],
        'cross_val_score': cv_scores.mean(),
        'cross_val_std': cv_scores.std(),
        'class_priors': class_priors.tolist(),
        'feature_means_died': best_model.theta_[0].tolist(),
        'feature_means_survived': best_model.theta_[1].tolist()
    }
    
    pd.Series(results_summary).to_json('results_summary.json', indent=2)
    
    print("[OK] Model and results saved!")
    print()
    print("[TARGET] Summary:")
    print(f"  Final Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Training Time: {best_result['training_time']:.4f}s")
    print(f"  Prediction Speed: {1/pred_time:.0f} predictions/second")
    print(f"  Cross-Validation: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print()
    print("[FOLDER] Generated Files:")
    print("  - naive_bayes_model.pkl (trained model)")
    print("  - feature_names.pkl (feature information)")
    print("  - results_summary.json (performance metrics)")
    print("  - *.png (visualization plots)")
    print()
    print("[START] Next: Run 'jupyter notebook explanation.ipynb' for detailed analysis!")

if __name__ == "__main__":
    train_naive_bayes()