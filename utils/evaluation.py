"""
Evaluation utilities for classification models.
Provides comprehensive metrics and visualization functions.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import learning_curve, validation_curve
import time
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Comprehensive model evaluation class."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.results = {}
        
    def evaluate_classification(self, y_true, y_pred, y_proba=None, 
                               class_names=None, pos_label=1):
        """
        Comprehensive classification evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            class_names: Names of classes (optional)
            pos_label: Positive class label for binary classification
            
        Returns:
            Dictionary with all metrics
        """
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Handle multi-class vs binary classification
        is_binary = len(np.unique(y_true)) == 2
        
        if is_binary:
            precision = precision_score(y_true, y_pred, pos_label=pos_label)
            recall = recall_score(y_true, y_pred, pos_label=pos_label)
            f1 = f1_score(y_true, y_pred, pos_label=pos_label)
            
            # ROC-AUC (if probabilities available)
            auc = None
            if y_proba is not None:
                if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                    auc = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    auc = roc_auc_score(y_true, y_proba)
        else:
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')
            
            # Multi-class ROC-AUC
            auc = None
            if y_proba is not None:
                try:
                    auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                except ValueError:
                    auc = None
        
        # Store results
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc,
            'classification_report': classification_report(y_true, y_pred, 
                                                         target_names=class_names),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        self.results.update(metrics)
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, 
                             figsize=(8, 6), normalize=False):
        """Plot confusion matrix with proper formatting."""
        
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', 
                   cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title(f'{self.model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        return plt.gcf()
    
    def plot_roc_curve(self, y_true, y_proba, figsize=(8, 6)):
        """Plot ROC curve for binary classification."""
        
        if len(np.unique(y_true)) != 2:
            print("ROC curve only available for binary classification")
            return None
            
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]  # Use positive class probability
            
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=figsize)
        plt.plot(fpr, tpr, label=f'{self.model_name} (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_precision_recall_curve(self, y_true, y_proba, figsize=(8, 6)):
        """Plot Precision-Recall curve for binary classification."""
        
        if len(np.unique(y_true)) != 2:
            print("PR curve only available for binary classification")
            return None
            
        if y_proba.ndim > 1:
            y_proba = y_proba[:, 1]  # Use positive class probability
            
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        plt.figure(figsize=figsize)
        plt.plot(recall, precision, label=f'{self.model_name} (AP = {avg_precision:.3f})', linewidth=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_learning_curve(self, estimator, X, y, cv=5, figsize=(10, 6)):
        """Plot learning curve to diagnose bias/variance."""
        
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=cv, train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy', n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=figsize)
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                        alpha=0.2, color='blue')
        
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                        alpha=0.2, color='red')
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title(f'{self.model_name} - Learning Curve')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def compare_models_metrics(self, evaluators: List['ModelEvaluator'], 
                              metrics=['accuracy', 'precision', 'recall', 'f1_score'],
                              figsize=(12, 8)):
        """Compare multiple models across different metrics."""
        
        model_names = [eval.model_name for eval in evaluators]
        
        # Create comparison dataframe
        comparison_data = []
        for evaluator in evaluators:
            row = {'Model': evaluator.model_name}
            for metric in metrics:
                row[metric.title()] = evaluator.results.get(metric, 0)
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            if i < len(axes):
                ax = axes[i]
                df.plot(x='Model', y=metric.title(), kind='bar', ax=ax, 
                       color=plt.cm.viridis(i/len(metrics)))
                ax.set_title(f'{metric.title()} Comparison')
                ax.set_xlabel('Model')
                ax.set_ylabel(metric.title())
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, df
    
    def print_detailed_report(self):
        """Print comprehensive evaluation report."""
        
        print(f"\n{'='*50}")
        print(f"DETAILED EVALUATION REPORT: {self.model_name}")
        print(f"{'='*50}")
        
        if 'accuracy' in self.results:
            print(f"\n[CHART] PERFORMANCE METRICS:")
            print(f"  Accuracy:  {self.results['accuracy']:.4f}")
            print(f"  Precision: {self.results['precision']:.4f}")
            print(f"  Recall:    {self.results['recall']:.4f}")
            print(f"  F1-Score:  {self.results['f1_score']:.4f}")
            
            if self.results.get('auc_score'):
                print(f"  AUC Score: {self.results['auc_score']:.4f}")
        
        if 'classification_report' in self.results:
            print(f"\n[LIST] CLASSIFICATION REPORT:")
            print(self.results['classification_report'])
        
        if 'training_time' in self.results:
            print(f"\n[TIME] TIMING INFORMATION:")
            print(f"  Training Time: {self.results['training_time']:.2f} seconds")
            
        if 'prediction_time' in self.results:
            print(f"  Prediction Time: {self.results['prediction_time']:.4f} seconds")

def measure_training_time(func):
    """Decorator to measure training time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        training_time = end_time - start_time
        return result, training_time
    return wrapper

def measure_prediction_time(model, X_test, n_runs=5):
    """Measure average prediction time."""
    times = []
    for _ in range(n_runs):
        start_time = time.time()
        _ = model.predict(X_test)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return np.mean(times)

def create_performance_summary(evaluators: List[ModelEvaluator]) -> pd.DataFrame:
    """Create a summary dataframe of model performances."""
    
    summary_data = []
    for evaluator in evaluators:
        row = {
            'Model': evaluator.model_name,
            'Accuracy': evaluator.results.get('accuracy', 0),
            'Precision': evaluator.results.get('precision', 0),
            'Recall': evaluator.results.get('recall', 0),
            'F1-Score': evaluator.results.get('f1_score', 0),
            'AUC': evaluator.results.get('auc_score', 0),
            'Training_Time': evaluator.results.get('training_time', 0),
            'Prediction_Time': evaluator.results.get('prediction_time', 0)
        }
        summary_data.append(row)
    
    df = pd.DataFrame(summary_data)
    
    # Round numerical columns
    numeric_columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    for col in numeric_columns:
        df[col] = df[col].round(4)
    
    # Format time columns
    if 'Training_Time' in df.columns:
        df['Training_Time'] = df['Training_Time'].round(2)
    if 'Prediction_Time' in df.columns:
        df['Prediction_Time'] = df['Prediction_Time'].round(4)
    
    return df

def plot_algorithm_comparison(summary_df: pd.DataFrame, figsize=(15, 10)):
    """Create comprehensive comparison plots for all algorithms."""
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.ravel()
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Training_Time']
    colors = plt.cm.Set3(np.linspace(0, 1, len(summary_df)))
    
    for i, metric in enumerate(metrics):
        if i < len(axes) and metric in summary_df.columns:
            ax = axes[i]
            bars = ax.bar(summary_df['Model'], summary_df[metric], color=colors)
            ax.set_title(f'{metric} Comparison')
            ax.set_xlabel('Algorithm')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, summary_df[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Remove empty subplot
    if len(axes) > len(metrics):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    return fig