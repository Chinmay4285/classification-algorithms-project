"""
Data utility functions for loading and preprocessing datasets.
Optimized for NVIDIA RTX 3060 GPU (6GB VRAM).
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.datasets import fetch_openml
import requests
from pathlib import Path

# Set device for GPU optimization
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

def create_data_dir():
    """Create datasets directory if it doesn't exist."""
    data_dir = Path(__file__).parent.parent / 'datasets'
    data_dir.mkdir(exist_ok=True)
    return data_dir

def load_titanic_data():
    """
    Load Titanic dataset for binary classification.
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    data_dir = create_data_dir()
    titanic_path = data_dir / 'titanic.csv'
    
    if not titanic_path.exists():
        print("Downloading Titanic dataset...")
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        response = requests.get(url)
        with open(titanic_path, 'wb') as f:
            f.write(response.content)
        print("Titanic dataset downloaded!")
    
    # Load and preprocess
    df = pd.read_csv(titanic_path)
    
    # Feature engineering
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Create family size feature
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Extract title from name
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 
                                       'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    # Select features
    features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone', 'Title']
    X = df[features].copy()
    y = df['Survived'].copy()
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    le_embarked = LabelEncoder()
    le_title = LabelEncoder()
    
    X['Sex'] = le_sex.fit_transform(X['Sex'])
    X['Embarked'] = le_embarked.fit_transform(X['Embarked'])
    X['Title'] = le_title.fit_transform(X['Title'])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    feature_names = X.columns.tolist()
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, feature_names

def load_fashion_mnist(subset_size=10000):
    """
    Load Fashion-MNIST dataset for image classification.
    Reduced size for GPU memory constraints.
    Returns: X_train, X_test, y_train, y_test, class_names
    """
    print(f"Loading Fashion-MNIST dataset (subset size: {subset_size})...")
    
    # Load Fashion-MNIST
    fashion_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
    X, y = fashion_mnist.data, fashion_mnist.target.astype(int)
    
    # Subset for memory efficiency
    if subset_size < len(X):
        indices = np.random.RandomState(42).choice(len(X), subset_size, replace=False)
        X, y = X[indices], y[indices]
    
    # Normalize pixel values
    X = X.astype(np.float32) / 255.0
    
    # Reshape for CNN if needed (28x28 images)
    X_reshaped = X.reshape(-1, 28, 28, 1)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y, test_size=0.2, random_state=42, stratify=y
    )
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    return X_train, X_test, y_train, y_test, class_names

def load_imdb_data(max_samples=5000):
    """
    Load IMDB movie reviews for text classification.
    Reduced size for GPU memory constraints.
    Returns: X_train, X_test, y_train, y_test
    """
    print(f"Loading IMDB dataset (max samples: {max_samples})...")
    
    try:
        from datasets import load_dataset
        dataset = load_dataset("imdb")
        
        # Get subset
        train_size = min(max_samples // 2, len(dataset['train']))
        test_size = min(max_samples // 2, len(dataset['test']))
        
        train_data = dataset['train'].select(range(train_size))
        test_data = dataset['test'].select(range(test_size))
        
        X_train = [item['text'] for item in train_data]
        y_train = [item['label'] for item in train_data]
        X_test = [item['text'] for item in test_data]
        y_test = [item['label'] for item in test_data]
        
        return X_train, X_test, y_train, y_test
        
    except ImportError:
        print("Using alternative IMDB data loading method...")
        # Fallback method using sklearn
        from sklearn.datasets import fetch_20newsgroups
        categories = ['alt.atheism', 'soc.religion.christian']
        
        train_data = fetch_20newsgroups(subset='train', categories=categories,
                                      remove=('headers', 'footers', 'quotes'))
        test_data = fetch_20newsgroups(subset='test', categories=categories,
                                     remove=('headers', 'footers', 'quotes'))
        
        # Limit samples
        train_size = min(max_samples // 2, len(train_data.data))
        test_size = min(max_samples // 2, len(test_data.data))
        
        X_train = train_data.data[:train_size]
        y_train = train_data.target[:train_size]
        X_test = test_data.data[:test_size]
        y_test = test_data.target[:test_size]
        
        return X_train, X_test, y_train, y_test

def load_spam_email_data():
    """
    Load Spam Email dataset for Naive Bayes (ideal for text classification).
    Returns: X_train, X_test, y_train, y_test
    """
    data_dir = create_data_dir()
    spam_path = data_dir / 'spam_email.csv'
    
    if not spam_path.exists():
        print("Downloading Spam Email dataset...")
        url = "https://raw.githubusercontent.com/mohitgupta-omg/Kaggle-SMS-Spam-Collection-Dataset-/master/spam.csv"
        try:
            response = requests.get(url)
            with open(spam_path, 'wb') as f:
                f.write(response.content)
            print("Spam Email dataset downloaded!")
        except Exception as e:
            print(f"Failed to download spam dataset: {e}")
            # Fallback to a simpler method
            return load_imdb_data(max_samples=3000)
    
    # Load and preprocess
    try:
        df = pd.read_csv(spam_path, encoding='latin-1')
        df = df[['v1', 'v2']]  # Keep only label and text columns
        df.columns = ['label', 'text']
        
        # Convert labels to binary
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        
        X = df['text'].values
        y = df['label'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train.tolist(), X_test.tolist(), y_train, y_test
        
    except Exception as e:
        print(f"Error loading spam dataset: {e}")
        return load_imdb_data(max_samples=3000)

def load_iris_data():
    """
    Load Iris dataset for KNN (classic dataset, ideal for distance-based algorithms).
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    print("Loading Iris dataset...")
    from sklearn.datasets import load_iris
    
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Scale features (important for KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    feature_names = iris.feature_names
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names

def load_mushroom_data():
    """
    Load Mushroom dataset for Decision Tree (categorical features, interpretable rules).
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    data_dir = create_data_dir()
    mushroom_path = data_dir / 'mushroom.csv'
    
    if not mushroom_path.exists():
        print("Downloading Mushroom dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
        try:
            response = requests.get(url)
            with open(mushroom_path, 'wb') as f:
                f.write(response.content)
            print("Mushroom dataset downloaded!")
        except Exception as e:
            print(f"Failed to download mushroom dataset: {e}")
            # Fallback to Titanic
            return load_titanic_data()
    
    # Column names for mushroom dataset
    columns = [
        'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
        'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
        'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
        'stalk-surface-below-ring', 'stalk-color-above-ring',
        'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
        'ring-type', 'spore-print-color', 'population', 'habitat'
    ]
    
    try:
        # Load data
        df = pd.read_csv(mushroom_path, names=columns)
        
        # Encode target
        df['class'] = df['class'].map({'e': 0, 'p': 1})  # edible=0, poisonous=1
        
        # Encode categorical features
        feature_columns = columns[1:]  # All except class
        for col in feature_columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        
        X = df[feature_columns].values
        y = df['class'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, feature_columns
        
    except Exception as e:
        print(f"Error loading mushroom dataset: {e}")
        return load_titanic_data()

def load_forest_cover_data(max_samples=50000):
    """
    Load Forest Cover Type dataset for Random Forest (large dataset, many features).
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    print(f"Loading Forest Cover Type dataset (max samples: {max_samples})...")
    
    try:
        from sklearn.datasets import fetch_covtype
        covtype = fetch_covtype()
        
        # Subsample for memory efficiency
        if max_samples < len(covtype.data):
            indices = np.random.RandomState(42).choice(len(covtype.data), max_samples, replace=False)
            X = covtype.data[indices]
            y = covtype.target[indices]
        else:
            X = covtype.data
            y = covtype.target
        
        # Convert to binary classification (simplify from 7 classes)
        # Class 1 vs All others
        y = (y == 1).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_names
        
    except Exception as e:
        print(f"Error loading forest cover dataset: {e}")
        # Fallback to Fashion-MNIST
        return load_fashion_mnist(subset_size=max_samples//10)

def load_adult_income_data():
    """
    Load Adult Income dataset for SVM (mixed features, margin-based classification).
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    data_dir = create_data_dir()
    adult_path = data_dir / 'adult.csv'
    
    if not adult_path.exists():
        print("Downloading Adult Income dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        try:
            response = requests.get(url)
            with open(adult_path, 'wb') as f:
                f.write(response.content)
            print("Adult Income dataset downloaded!")
        except Exception as e:
            print(f"Failed to download adult dataset: {e}")
            return load_wine_quality_data()
    
    # Column names
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    try:
        # Load data
        df = pd.read_csv(adult_path, names=columns, na_values=' ?', skipinitialspace=True)
        
        # Drop rows with missing values
        df = df.dropna()
        
        # Encode target
        df['income'] = (df['income'] == '>50K').astype(int)
        
        # Select features (mix of numerical and categorical)
        numerical_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 
                            'capital-loss', 'hours-per-week']
        categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 
                              'relationship', 'race', 'sex', 'native-country']
        
        # Encode categorical features
        for col in categorical_features:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
        
        feature_columns = numerical_features + categorical_features
        X = df[feature_columns].values
        y = df['income'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features (important for SVM)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns
        
    except Exception as e:
        print(f"Error loading adult dataset: {e}")
        return load_wine_quality_data()

def load_bike_sharing_data():
    """
    Load Bike Sharing dataset for Gradient Boosting (temporal patterns, regression to classification).
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    data_dir = create_data_dir()
    bike_path = data_dir / 'bike_sharing.csv'
    
    if not bike_path.exists():
        print("Downloading Bike Sharing dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
        try:
            response = requests.get(url)
            import zipfile
            import io
            
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                # Extract day.csv
                with z.open('day.csv') as f:
                    with open(bike_path, 'wb') as out_f:
                        out_f.write(f.read())
            print("Bike Sharing dataset downloaded!")
        except Exception as e:
            print(f"Failed to download bike sharing dataset: {e}")
            return load_wine_quality_data()
    
    try:
        # Load data
        df = pd.read_csv(bike_path)
        
        # Select relevant features
        features = ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday',
                   'weathersit', 'temp', 'atemp', 'hum', 'windspeed']
        
        X = df[features].values
        
        # Convert regression to binary classification
        # High demand (above median) vs Low demand
        y = (df['cnt'] > df['cnt'].median()).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test, features
        
    except Exception as e:
        print(f"Error loading bike sharing dataset: {e}")
        return load_wine_quality_data()

def load_credit_card_fraud_data(max_samples=50000):
    """
    Load Credit Card Fraud dataset for XGBoost/LightGBM/CatBoost (imbalanced, large dataset).
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    print(f"Loading Credit Card Fraud dataset (max samples: {max_samples})...")
    
    data_dir = create_data_dir()
    fraud_path = data_dir / 'creditcard.csv'
    
    if not fraud_path.exists():
        print("Credit Card Fraud dataset not found. Using Wine Quality dataset as fallback...")
        print("To use the fraud dataset, download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        return load_wine_quality_data()
    
    try:
        # Load data
        df = pd.read_csv(fraud_path)
        
        # Subsample if dataset is too large
        if max_samples < len(df):
            df = df.sample(n=max_samples, random_state=42)
        
        # Separate features and target
        X = df.drop(['Class'], axis=1).values
        y = df['Class'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        feature_names = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Time']
        
        return X_train, X_test, y_train, y_test, feature_names
        
    except Exception as e:
        print(f"Error loading credit card fraud dataset: {e}")
        return load_wine_quality_data()

def load_cifar10_subset(max_samples=8000):
    """
    Load CIFAR-10 subset for Neural Networks (image classification).
    Returns: X_train, X_test, y_train, y_test, class_names
    """
    print(f"Loading CIFAR-10 subset (max samples: {max_samples})...")
    
    try:
        from sklearn.datasets import fetch_openml
        
        # Load CIFAR-10 (this might take a while)
        cifar10 = fetch_openml('CIFAR_10', version=1, as_frame=False)
        X, y = cifar10.data, cifar10.target.astype(int)
        
        # Subsample for memory efficiency
        if max_samples < len(X):
            indices = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
            X, y = X[indices], y[indices]
        
        # Normalize pixel values
        X = X.astype(np.float32) / 255.0
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
        return X_train, X_test, y_train, y_test, class_names
        
    except Exception as e:
        print(f"Error loading CIFAR-10: {e}")
        # Fallback to Fashion-MNIST
        return load_fashion_mnist(subset_size=max_samples)

def load_news_category_data(max_samples=10000):
    """
    Load News Category dataset for Transformers (text classification with categories).
    Returns: X_train, X_test, y_train, y_test
    """
    print(f"Loading News Category dataset (max samples: {max_samples})...")
    
    data_dir = create_data_dir()
    news_path = data_dir / 'news_category.json'
    
    if not news_path.exists():
        print("Downloading News Category dataset...")
        url = "https://raw.githubusercontent.com/rmisra/News-Category-Dataset/master/News_Category_Dataset_v3.json"
        try:
            response = requests.get(url)
            with open(news_path, 'wb') as f:
                f.write(response.content)
            print("News Category dataset downloaded!")
        except Exception as e:
            print(f"Failed to download news category dataset: {e}")
            return load_imdb_data(max_samples)
    
    try:
        import json
        
        # Load data
        data = []
        with open(news_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        
        df = pd.DataFrame(data)
        
        # Select top categories for binary classification
        top_categories = df['category'].value_counts().head(2).index
        df_filtered = df[df['category'].isin(top_categories)]
        
        # Subsample if needed
        if max_samples < len(df_filtered):
            df_filtered = df_filtered.sample(n=max_samples, random_state=42)
        
        # Combine headline and description for text
        X = (df_filtered['headline'] + ' ' + df_filtered['short_description']).tolist()
        
        # Binary classification
        y = (df_filtered['category'] == top_categories[0]).astype(int).values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"Error loading news category dataset: {e}")
        return load_imdb_data(max_samples)

def load_wine_quality_data():
    """
    Load Wine Quality dataset for multi-class classification.
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    print("Loading Wine Quality dataset...")
    
    data_dir = create_data_dir()
    wine_path = data_dir / 'wine_quality.csv'
    
    if not wine_path.exists():
        print("Downloading Wine Quality dataset...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
        response = requests.get(url)
        with open(wine_path, 'wb') as f:
            f.write(response.content)
        print("Wine Quality dataset downloaded!")
    
    # Load data
    df = pd.read_csv(wine_path, delimiter=';')
    
    # Prepare features and target
    X = df.drop('quality', axis=1)
    y = df['quality']
    
    # Convert to binary classification (quality >= 6 is good)
    y = (y >= 6).astype(int)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    feature_names = X.columns.tolist()
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, feature_names

def prepare_text_data(texts, max_length=512):
    """
    Prepare text data for NLP models.
    Args:
        texts: List of text strings
        max_length: Maximum sequence length
    Returns:
        Preprocessed text data
    """
    # Basic preprocessing
    processed_texts = []
    for text in texts:
        # Convert to lowercase and limit length
        text = str(text).lower()[:max_length]
        processed_texts.append(text)
    
    return processed_texts

def get_gpu_memory_info():
    """
    Get GPU memory information for optimization.
    Returns: Dict with GPU memory stats
    """
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        
        return {
            'device_name': props.name,
            'total_memory_gb': props.total_memory / (1024**3),
            'allocated_gb': allocated / (1024**3),
            'reserved_gb': reserved / (1024**3),
            'free_gb': (props.total_memory - reserved) / (1024**3)
        }
    else:
        return {'message': 'CUDA not available'}

def optimize_batch_size_for_gpu(base_batch_size=32, model_size_mb=100):
    """
    Optimize batch size based on available GPU memory.
    Args:
        base_batch_size: Starting batch size
        model_size_mb: Estimated model size in MB
    Returns:
        Optimized batch size
    """
    if not torch.cuda.is_available():
        return min(base_batch_size, 16)  # Conservative for CPU
    
    gpu_info = get_gpu_memory_info()
    free_memory_mb = gpu_info['free_gb'] * 1024
    
    # Reserve 1GB for system
    available_mb = free_memory_mb - 1024
    
    # Estimate memory per sample (rough approximation)
    memory_per_sample = model_size_mb / 4  # Rough estimate
    
    # Calculate optimal batch size
    optimal_batch = max(1, int(available_mb / memory_per_sample))
    
    return min(optimal_batch, base_batch_size * 4)  # Cap at 4x base