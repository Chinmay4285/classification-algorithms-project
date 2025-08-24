# Naïve Bayes Classifier

## [TARGET] Algorithm Overview

Naïve Bayes is a probabilistic classifier based on **Bayes' theorem** with a "naïve" assumption of independence between features. Despite this strong assumption, it often performs surprisingly well in practice, especially for text classification and spam detection.

## [MATH] How It Works

### Mathematical Foundation

1. **Bayes' Theorem**:
   ```
   P(Class|Features) = P(Features|Class) × P(Class) / P(Features)
   ```

2. **Naïve Independence Assumption**:
   ```
   P(x₁,x₂,...,xₙ|Class) = P(x₁|Class) × P(x₂|Class) × ... × P(xₙ|Class)
   ```

3. **Classification Rule**:
   ```
   ŷ = argmax P(Class) × ∏ P(feature_i|Class)
   ```

### Key Variants

- **Gaussian NB**: Assumes features follow normal distribution
- **Multinomial NB**: Good for discrete features (word counts)
- **Bernoulli NB**: Good for binary features (presence/absence)

## [START] Advantages

- **Fast Training**: Requires only one pass through training data
- **Fast Prediction**: Simple probability calculations
- **Small Dataset Friendly**: Works well with limited training data
- **Multiclass Natural**: Handles multiple classes naturally
- **Interpretable**: Probabilities have clear meaning
- **Robust to Noise**: Less affected by irrelevant features

## [WARNING] Limitations

- **Independence Assumption**: Assumes features are independent (rarely true)
- **Feature Correlation**: Performs poorly when features are highly correlated
- **Continuous Features**: Gaussian assumption may not hold
- **Zero Probability**: Can assign zero probability to unseen feature combinations
- **Calibration**: Probability estimates can be poor

## [CHART] Dataset Used

**Titanic Survival Dataset**
- **Type**: Binary Classification (Survived: 0/1)
- **Features**: 8 engineered features
  - `Pclass`: Passenger class (1, 2, 3)
  - `Sex`: Gender (encoded)
  - `Age`: Age in years
  - `Fare`: Ticket fare
  - `Embarked`: Port of embarkation (encoded)
  - `FamilySize`: Family size
  - `IsAlone`: Traveling alone (0/1)  
  - `Title`: Extracted title from name (encoded)
- **Samples**: ~891 passengers
- **Target**: Binary (Survived: 1, Died: 0)
- **Variant Used**: Gaussian Naïve Bayes (continuous features)

## [PARAM] Implementation Details

### Hyperparameters
- **var_smoothing**: Smoothing parameter to handle numerical instability (default: 1e-9)
- **priors**: Prior probabilities of classes (default: fitted from data)

### GPU Optimization
- Uses NumPy's optimized operations
- Efficient probability calculations
- Memory-friendly for RTX 3060 (6GB VRAM)
- Fast matrix operations for batch predictions

## [INFO] Expected Performance

- **Accuracy**: ~75-80%
- **Training Time**: < 0.1 seconds
- **Prediction Time**: < 0.01 seconds
- **Memory Usage**: Minimal

## 🏃‍♂️ Quick Start

```bash
cd 02_naive_bayes
python train.py
```

Or explore the interactive notebook:
```bash
jupyter notebook explanation.ipynb
```

## 📚 Use Cases

### When to Use Naïve Bayes
- **Text Classification**: Spam detection, sentiment analysis
- **Small Datasets**: Limited training data available
- **Baseline Model**: Quick first model to establish benchmark
- **Real-time Systems**: Need extremely fast predictions
- **Multiclass Problems**: Natural handling of multiple classes
- **Feature Independence**: When features are relatively independent

### When NOT to Use
- **Highly Correlated Features**: Strong feature dependencies exist
- **Continuous Complex Data**: Images, audio signals
- **Need Calibrated Probabilities**: Probability estimates are important
- **Complex Relationships**: Non-linear feature interactions

## [TOOL] Example Usage

```python
from sklearn.naive_bayes import GaussianNB
from utils.data_utils import load_titanic_data
from utils.evaluation import ModelEvaluator

# Load data
X_train, X_test, y_train, y_test, feature_names = load_titanic_data()

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Evaluate
evaluator = ModelEvaluator("Naïve Bayes")
metrics = evaluator.evaluate_classification(y_test, y_pred, y_proba)
```

## [CHART] Sample Input/Output

### Input Features (First 3 samples):
```
Pclass  Sex  Age   Fare    Embarked  FamilySize  IsAlone  Title
3       0    22.0  7.25    2         2           0        1
1       1    38.0  71.28   0         2           0        3  
3       1    26.0  7.925   2         1           1        2
```

### Model Output:
```
Predictions:     [0, 1, 0]
Probabilities:   [[0.85, 0.15], [0.25, 0.75], [0.82, 0.18]]
Interpretation:  [Died (85%), Survived (75%), Died (82%)]
```

### Feature Statistics per Class:
```
Feature: Sex
  Class 0 (Died):     Mean=0.15, Std=0.36
  Class 1 (Survived): Mean=0.68, Std=0.47

Feature: Pclass  
  Class 0 (Died):     Mean=2.53, Std=0.71
  Class 1 (Survived): Mean=1.95, Std=0.86
```

## 🤔 Why "Naïve"?

The algorithm is called "naïve" because it assumes features are **conditionally independent** given the class. For example, it assumes:

```
P(Age=30, Sex=Female | Survived) = P(Age=30 | Survived) × P(Sex=Female | Survived)
```

This is often unrealistic (age and fare might be correlated), but the algorithm works well despite this simplification!

## [CHART] Mathematical Intuition

### For Gaussian Naïve Bayes:
1. **Calculate class priors**: P(Survived), P(Died)
2. **For each feature, calculate likelihood**:
   ```
   P(feature_value | class) = (1/√(2πσ²)) × exp(-((x-μ)²)/(2σ²))
   ```
3. **Multiply everything**:
   ```
   P(Survived | features) ∝ P(Survived) × ∏ P(feature_i | Survived)
   ```

### Smoothing
- **Laplace Smoothing**: Add small constant to avoid zero probabilities
- **Gaussian Smoothing**: Add small variance to handle numerical stability