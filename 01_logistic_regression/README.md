# Logistic Regression

## [TARGET] Algorithm Overview

Logistic Regression is a linear classifier that uses the logistic function (sigmoid) to model the probability of binary or multi-class classification problems. Despite its name, it's a classification algorithm, not a regression algorithm.

## [MATH] How It Works

### Mathematical Foundation

1. **Linear Combination**: Creates a linear combination of features:
   ```
   z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
   ```

2. **Sigmoid Function**: Maps the linear output to probabilities:
   ```
   p = 1 / (1 + e^(-z))
   ```

3. **Decision Rule**: Classifies based on probability threshold (typically 0.5):
   ```
   ŷ = 1 if p ≥ 0.5, else 0
   ```

### Key Characteristics

- **Linear Decision Boundary**: Creates a linear separation between classes
- **Probabilistic Output**: Provides class probabilities, not just predictions
- **No Assumptions**: Doesn't assume normal distribution of features
- **Maximum Likelihood**: Uses MLE for parameter estimation

## [START] Advantages

- **Interpretable**: Coefficients indicate feature importance and direction
- **Fast**: Efficient training and prediction
- **No Tuning**: Few hyperparameters to tune
- **Probabilistic**: Outputs meaningful probabilities
- **Robust**: Less prone to overfitting with regularization

## [WARNING] Limitations

- **Linear Boundary**: Can't capture non-linear relationships
- **Feature Scaling**: Sensitive to feature scales
- **Outliers**: Can be affected by extreme values
- **Feature Independence**: Assumes features are independent
- **Large Datasets**: May need regularization for high-dimensional data

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

## [PARAM] Implementation Details

### Hyperparameters
- **C**: Inverse of regularization strength (default: 1.0)
- **penalty**: Regularization type ('l1', 'l2', 'elasticnet')
- **solver**: Optimization algorithm ('liblinear', 'lbfgs', 'sag')
- **max_iter**: Maximum iterations (default: 1000)

### GPU Optimization
- Uses scikit-learn's optimized solvers
- Efficient matrix operations with NumPy
- Memory-friendly for RTX 3060 (6GB VRAM)

## [INFO] Expected Performance

- **Accuracy**: ~79-82%
- **Training Time**: < 1 second
- **Prediction Time**: < 0.01 seconds
- **Memory Usage**: Minimal

## 🏃‍♂️ Quick Start

```bash
cd 01_logistic_regression
python train.py
```

Or explore the interactive notebook:
```bash
jupyter notebook explanation.ipynb
```

## 📚 Use Cases

### When to Use Logistic Regression
- **Baseline Model**: Start with logistic regression
- **Interpretability**: Need to understand feature importance
- **Limited Data**: Works well with small datasets
- **Linear Relationships**: Features have linear relationship with log-odds
- **Real-time Predictions**: Need fast inference

### When NOT to Use
- **Complex Patterns**: Non-linear relationships in data
- **Feature Interactions**: Complex feature interactions exist
- **Image/Text**: Raw images or unstructured text (use after feature extraction)
- **Very High Dimensions**: Many features without regularization

## [TOOL] Example Usage

```python
from sklearn.linear_model import LogisticRegression
from utils.data_utils import load_titanic_data
from utils.evaluation import ModelEvaluator

# Load data
X_train, X_test, y_train, y_test, feature_names = load_titanic_data()

# Train model
model = LogisticRegression(C=1.0, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Evaluate
evaluator = ModelEvaluator("Logistic Regression")
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
Probabilities:   [[0.91, 0.09], [0.18, 0.82], [0.88, 0.12]]
Interpretation:  [Died (91%), Survived (82%), Died (88%)]
```

### Feature Importance (Coefficients):
```
Sex (Female):    +2.52  (Strong positive effect on survival)
Pclass:         -1.14   (Higher class increases survival)
Age:            -0.04   (Slight negative effect)
Fare:           +0.003  (Higher fare increases survival)
Title (Mrs):    +1.45   (Married women more likely to survive)
```