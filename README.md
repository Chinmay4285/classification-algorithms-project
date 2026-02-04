# Classification Algorithms Benchmark Suite

Systematic implementation and evaluation of 10 core classification algorithms with GPU optimization and standardized benchmarks on canonical datasets.

## Overview

Production-grade implementations demonstrating algorithm fundamentals, performance characteristics, and computational trade-offs across classical ML and deep learning approaches:

1. **Logistic Regression** - Baseline linear classifier with probabilistic outputs
2. **Naive Bayes** - Fast probabilistic classifier for high-dimensional data
3. **K-Nearest Neighbors** - Instance-based learning with distance metrics
4. **Decision Tree** - Interpretable rule-based classification
5. **Random Forest** - Ensemble method with bagging and feature randomness
6. **Support Vector Machine** - Kernel-based margin optimization
7. **Gradient Boosting** - Sequential weak learner ensemble
8. **XGBoost/LightGBM/CatBoost** - Industrial-strength gradient boosting variants
9. **Neural Networks** - Multi-layer perceptrons with GPU acceleration
10. **Transformers** - Attention-based architectures for structured data

## Benchmark Results

**Titanic Dataset (891 samples, binary classification)**
| Algorithm | Accuracy | Notes |
|-----------|----------|-------|
| SVM | 83.2% | Best performer, RBF kernel |
| Random Forest | 82.1% | 100 estimators, max_depth=10 |
| KNN | 81.0% | k=7, Euclidean distance |
| Naive Bayes | 81.0% | Gaussian distribution |
| Decision Tree | 80.4% | max_depth=5, Gini criterion |
| Neural Network | 79.9% | 2 hidden layers [64, 32] |
| Logistic Regression | 79.3% | L2 regularization |

**Iris Dataset (150 samples, 3-class classification)**
| Algorithm | Accuracy | Notes |
|-----------|----------|-------|
| SVM | 93.3% | Linear kernel |
| Neural Network | 93.3% | Shallow architecture |
| Logistic Regression | 91.1% | Multi-class one-vs-rest |
| Naive Bayes | 91.1% | Well-suited for Gaussian features |
| Decision Tree | 91.1% | Low depth sufficient |
| KNN | 91.1% | k=5 |
| Random Forest | 88.9% | Minor overfitting |

*Results from 5-fold cross-validation. Training on NVIDIA RTX 3060 where applicable.*

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Chinmay4285/classification-algorithms-project.git
cd classification-algorithms-project
pip install -r requirements.txt

# Run individual algorithm
cd 06_svm && python train.py

# Compare all algorithms (generates CSV + visualizations)
python compare_algorithms.py
```

**Output artifacts:**
- `algorithm_comparison.csv` - Performance metrics table
- `comparison_results.png` - Accuracy/time plots across algorithms

## Repository Structure

```
classification-algorithms-project/
├── 01_logistic_regression/          # Baseline linear model
├── 02_naive_bayes/                  # Probabilistic classifier
├── 03_knn/                          # Distance-based learning
├── 04_decision_tree/                # Rule-based classifier
├── 05_random_forest/                # Bagging ensemble
├── 06_svm/                          # Kernel methods
├── 07_gradient_boosting/            # Boosting ensemble
├── 08_xgboost_lightgbm_catboost/   # Industrial boosting variants
├── 09_neural_networks/              # Deep learning (GPU)
├── 10_transformers/                 # Attention mechanisms
├── utils/
│   ├── data_utils.py               # Dataset loaders
│   ├── evaluation.py               # Metrics computation
│   └── visualization.py            # Plotting utilities
├── datasets/                        # Cached data files
├── compare_algorithms.py            # Benchmark orchestrator
└── requirements.txt
```

Each algorithm directory contains:
- `train.py` - Training and evaluation script
- `explanation.ipynb` - Algorithm theory and implementation details
- `README.md` - Algorithm-specific documentation
├── requirements.txt           # Python dependencies
└── README.md
```

## Technical Stack

- **Core ML**: scikit-learn 1.3+, XGBoost, LightGBM, CatBoost
- **Deep Learning**: PyTorch 2.0+ with CUDA 11.8+
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Hardware**: Optimized for NVIDIA RTX 3060 (6GB VRAM)

## Key Features

- **Standardized evaluation**: Consistent train/test splits, cross-validation, and metrics
- **GPU acceleration**: CUDA-enabled neural networks and compatible boosting libraries
- **Comparative analysis**: Side-by-side algorithm performance with runtime profiling
- **Educational value**: Clean implementations demonstrating core algorithm principles
- **Production patterns**: Error handling, logging, model serialization

## Usage Examples

```python
# Load datasets programmatically
from utils.data_utils import load_titanic_data, load_iris_data

X_train, X_test, y_train, y_test, feature_names = load_titanic_data()

# Train specific algorithm
from sklearn.svm import SVC
model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train, y_train)

# Evaluate
from utils.evaluation import evaluate_model
metrics = evaluate_model(model, X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.3f}")
```

## Limitations & Future Work

- **Datasets**: Limited to small benchmark datasets; extend to large-scale problems
- **Hyperparameter tuning**: Manual tuning; implement automated search (Optuna/Ray Tune)
- **Distributed training**: Single-GPU only; add multi-GPU support
- **Model interpretability**: Add SHAP/LIME for black-box models
- **Streaming inference**: Batch-only; add online learning capabilities