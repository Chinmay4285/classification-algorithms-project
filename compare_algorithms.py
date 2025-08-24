"""
Simple Algorithm Comparison Script
Core functionality only - no add-ons
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

from utils.data_utils import load_titanic_data, load_iris_data

def compare_algorithms():
    """Compare core algorithms on standard datasets."""
    
    print("Algorithm Comparison - Core Functionality")
    print("=" * 45)
    
    # Load datasets
    X_titanic_train, X_titanic_test, y_titanic_train, y_titanic_test, _ = load_titanic_data()
    X_iris_train, X_iris_test, y_iris_train, y_iris_test, _ = load_iris_data()
    
    datasets = {
        'Titanic': (X_titanic_train, X_titanic_test, y_titanic_train, y_titanic_test),
        'Iris': (X_iris_train, X_iris_test, y_iris_train, y_iris_test)
    }
    
    # Core algorithms
    algorithms = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(50,), random_state=42, max_iter=500)
    }
    
    results = []
    
    # Train and evaluate
    for dataset_name, (X_train, X_test, y_train, y_test) in datasets.items():
        print(f"\nDataset: {dataset_name}")
        print("-" * 20)
        
        for algo_name, model in algorithms.items():
            try:
                # Train
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                
                # Predict
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train, y_train, cv=3)
                
                results.append({
                    'Dataset': dataset_name,
                    'Algorithm': algo_name,
                    'Accuracy': accuracy,
                    'CV_Mean': cv_scores.mean(),
                    'Training_Time': training_time
                })
                
                print(f"{algo_name:18}: {accuracy:.3f} accuracy ({training_time:.3f}s)")
                
            except Exception as e:
                print(f"{algo_name:18}: Failed - {str(e)[:30]}...")
    
    # Create results summary
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv('algorithm_comparison.csv', index=False)
    
    # Simple visualization
    if len(results_df) > 0:
        plt.figure(figsize=(12, 6))
        
        # Accuracy by algorithm
        plt.subplot(1, 2, 1)
        avg_acc = results_df.groupby('Algorithm')['Accuracy'].mean().sort_values(ascending=True)
        plt.barh(range(len(avg_acc)), avg_acc.values)
        plt.yticks(range(len(avg_acc)), avg_acc.index)
        plt.xlabel('Average Accuracy')
        plt.title('Algorithm Performance')
        
        # Training time
        plt.subplot(1, 2, 2)
        avg_time = results_df.groupby('Algorithm')['Training_Time'].mean().sort_values(ascending=True)
        plt.barh(range(len(avg_time)), avg_time.values)
        plt.yticks(range(len(avg_time)), avg_time.index)
        plt.xlabel('Average Training Time (s)')
        plt.title('Training Speed')
        
        plt.tight_layout()
        plt.savefig('comparison_results.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    print(f"\nResults saved to: algorithm_comparison.csv")
    print("Visualization saved to: comparison_results.png")

if __name__ == "__main__":
    compare_algorithms()