# Machine Learning Classification Algorithms Project

A comprehensive implementation of 10 classification algorithms optimized for NVIDIA RTX 3060 GPU (6GB VRAM) with core functionality and performance comparisons.

## Project Overview

This project implements the most important classification algorithms in machine learning:

1. **Logistic Regression** - Linear probabilistic classifier
2. **Naive Bayes** - Probabilistic classifier based on Bayes' theorem  
3. **K-Nearest Neighbors (KNN)** - Instance-based learning algorithm
4. **Decision Tree** - Tree-based classification with interpretable rules
5. **Random Forest** - Ensemble of decision trees
6. **Support Vector Machine (SVM)** - Margin-maximizing classifier
7. **Gradient Boosting** - Sequential ensemble method
8. **XGBoost/LightGBM/CatBoost** - Advanced gradient boosting implementations
9. **Neural Networks** - Multi-layer perceptrons with GPU acceleration
10. **Transformer Models** - Modern deep learning architectures

## Performance Results

### Titanic Dataset (Binary Classification)
- **SVM**: 83.2% accuracy
- **Random Forest**: 82.1% accuracy  
- **KNN**: 81.0% accuracy
- **Naive Bayes**: 81.0% accuracy
- **Decision Tree**: 80.4% accuracy
- **Logistic Regression**: 79.3% accuracy
- **Neural Network**: 79.9% accuracy

### Iris Dataset (Multi-class Classification)  
- **SVM**: 93.3% accuracy
- **KNN**: 91.1% accuracy
- **Logistic Regression**: 91.1% accuracy
- **Naive Bayes**: 91.1% accuracy
- **Decision Tree**: 91.1% accuracy
- **Neural Network**: 93.3% accuracy
- **Random Forest**: 88.9% accuracy

## Quick Start

### Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA support
- 16GB+ RAM recommended

### Installation
```bash
git clone https://github.com/yourusername/classification-algorithms-project.git
cd classification-algorithms-project
pip install -r requirements.txt
```

### Running Algorithms

**Individual Algorithm Training:**
```bash
# Example: Run KNN algorithm
cd 03_knn
python train.py

# Example: Run Logistic Regression  
cd 01_logistic_regression
python train.py
```

**Compare All Algorithms:**
```bash
# Run comprehensive comparison
python compare_algorithms.py
```

This generates:
- `algorithm_comparison.csv` - Performance metrics
- `comparison_results.png` - Visualization charts

## Project Structure

```
classification_project/
├── 01_logistic_regression/
│   ├── train.py                 # Training script
│   ├── explanation.ipynb        # Interactive notebook
│   └── README.md               # Algorithm explanation
├── 02_naive_bayes/
├── 03_knn/
├── 04_decision_tree/
├── 05_random_forest/
├── 06_svm/
├── 07_gradient_boosting/
├── 08_xgboost_lightgbm_catboost/
├── 09_neural_networks/
├── 10_transformers/
├── utils/
│   ├── data_utils.py           # Dataset loading utilities
│   ├── evaluation.py           # Model evaluation tools
│   └── visualization.py        # Plotting functions
├── datasets/
│   └── titanic.csv            # Cached datasets
├── compare_algorithms.py       # Master comparison script
├── requirements.txt           # Python dependencies
└── README.md
```

## Datasets Used

- **Titanic Survival** - Binary classification with passenger data
- **Iris Flowers** - Multi-class classification with flower measurements
- Additional datasets loaded automatically for each algorithm

## GPU Optimization

Optimized for NVIDIA RTX 3060 (6GB VRAM):
- GPU memory monitoring
- CUDA acceleration where supported
- Efficient batch processing
- Memory-conscious model configurations

## Key Features

- **Core Functionality**: Clean, efficient implementations without unnecessary add-ons
- **Performance Metrics**: Accuracy, Precision, Recall, F1-Score, Training Time
- **Cross-Validation**: 3-5 fold validation for robust evaluation
- **Visualization**: Training curves, confusion matrices, performance comparisons
- **GPU Support**: CUDA detection and utilization
- **Error Handling**: Robust exception handling and informative logging

## Hardware Requirements

- **GPU**: NVIDIA RTX 3060 (6GB VRAM) or better
- **RAM**: 16GB+ recommended  
- **Storage**: 2GB+ free space
- **CUDA**: 11.0+ with compatible PyTorch

## Usage Examples

**Train a specific algorithm:**
```python
cd 03_knn
python train.py
```

**Compare all algorithms:**
```python  
python compare_algorithms.py
```

**Load datasets programmatically:**
```python
from utils.data_utils import load_titanic_data, load_iris_data

X_train, X_test, y_train, y_test, features = load_titanic_data()
X_train, X_test, y_train, y_test, features = load_iris_data()
```

## Contributing

Feel free to add new algorithms, datasets, or optimizations while maintaining the project's core functionality focus.

## License

Open source - feel free to use for educational and research purposes.