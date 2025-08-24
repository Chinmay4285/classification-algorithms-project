"""
Visualization utilities for machine learning classification project.
Provides functions for data exploration and model interpretation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_data_distribution(X, y, feature_names=None, figsize=(15, 10), 
                          max_features=8, title="Data Distribution"):
    """
    Plot distribution of features colored by target variable.
    
    Args:
        X: Feature matrix
        y: Target variable
        feature_names: Names of features
        figsize: Figure size
        max_features: Maximum number of features to plot
        title: Plot title
    """
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    # Limit features for readability
    n_features = min(len(feature_names), max_features)
    
    # Create subplot grid
    n_rows = (n_features + 3) // 4
    n_cols = min(4, n_features)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(n_features):
        ax = axes[i] if n_features > 1 else axes[0]
        
        # Create DataFrame for seaborn
        df_plot = pd.DataFrame({
            'feature': X[:, i],
            'target': y
        })
        
        # Plot histogram with target coloring
        for target_val in np.unique(y):
            subset = df_plot[df_plot['target'] == target_val]
            ax.hist(subset['feature'], alpha=0.7, label=f'Class {target_val}', 
                   bins=30, density=True)
        
        ax.set_title(f'{feature_names[i]}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig

def plot_correlation_matrix(X, feature_names=None, figsize=(10, 8)):
    """Plot correlation matrix of features."""
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    df = pd.DataFrame(X, columns=feature_names)
    corr_matrix = df.corr()
    
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    return plt.gcf()

def plot_pca_visualization(X, y, feature_names=None, figsize=(12, 5)):
    """
    Create PCA visualization of the data.
    
    Args:
        X: Feature matrix
        y: Target variable
        feature_names: Names of features
        figsize: Figure size
    """
    
    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 2D PCA
    scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[0].set_title('PCA - First Two Components')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0])
    
    # Plot explained variance ratio
    n_components = min(10, len(pca.explained_variance_ratio_))
    axes[1].bar(range(1, n_components + 1), pca.explained_variance_ratio_[:n_components])
    axes[1].set_xlabel('Principal Component')
    axes[1].set_ylabel('Explained Variance Ratio')
    axes[1].set_title('Explained Variance by Component')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_tsne_visualization(X, y, perplexity=30, figsize=(10, 8)):
    """
    Create t-SNE visualization of the data.
    
    Args:
        X: Feature matrix
        y: Target variable
        perplexity: t-SNE perplexity parameter
        figsize: Figure size
    """
    
    # Limit samples for t-SNE performance
    if X.shape[0] > 1000:
        indices = np.random.choice(X.shape[0], 1000, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]
    else:
        X_sample = X
        y_sample = y
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X_sample)
    
    plt.figure(figsize=figsize)
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, cmap='viridis', alpha=0.7)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title(f't-SNE Visualization (perplexity={perplexity})')
    plt.colorbar(scatter)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def plot_class_distribution(y, class_names=None, figsize=(10, 6)):
    """Plot distribution of target classes."""
    
    if class_names is None:
        class_names = [f'Class {i}' for i in np.unique(y)]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Count plot
    unique_classes, counts = np.unique(y, return_counts=True)
    bars = axes[0].bar([class_names[i] for i in unique_classes], counts)
    axes[0].set_title('Class Distribution')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].grid(True, alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{count}', ha='center', va='bottom')
    
    # Pie chart
    axes[1].pie(counts, labels=[class_names[i] for i in unique_classes], 
               autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Class Distribution (Percentage)')
    
    plt.tight_layout()
    return fig

def plot_feature_importance(importance_scores, feature_names=None, 
                          top_k=15, figsize=(10, 8), title="Feature Importance"):
    """
    Plot feature importance scores.
    
    Args:
        importance_scores: Array of importance scores
        feature_names: Names of features
        top_k: Number of top features to show
        figsize: Figure size
        title: Plot title
    """
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(importance_scores))]
    
    # Sort features by importance
    indices = np.argsort(importance_scores)[::-1]
    top_indices = indices[:top_k]
    
    plt.figure(figsize=figsize)
    bars = plt.barh(range(len(top_indices)), 
                   importance_scores[top_indices], 
                   color=plt.cm.viridis(np.linspace(0, 1, len(top_indices))))
    
    plt.yticks(range(len(top_indices)), 
              [feature_names[i] for i in top_indices])
    plt.xlabel('Importance Score')
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, importance_scores[top_indices])):
        width = bar.get_width()
        plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2.,
                f'{score:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    return plt.gcf()

def plot_decision_boundary_2d(X, y, model, resolution=100, figsize=(10, 8)):
    """
    Plot decision boundary for 2D data.
    
    Args:
        X: 2D feature matrix
        y: Target variable
        model: Trained model
        resolution: Grid resolution
        figsize: Figure size
    """
    
    if X.shape[1] != 2:
        # Use PCA to reduce to 2D
        pca = PCA(n_components=2)
        X = pca.fit_transform(X)
    
    # Create mesh grid
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))
    
    # Make predictions on mesh grid
    try:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=figsize)
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black')
        plt.xlabel('Feature 1' if X.shape[1] == 2 else 'PC1')
        plt.ylabel('Feature 2' if X.shape[1] == 2 else 'PC2')
        plt.title('Decision Boundary')
        plt.colorbar(scatter)
        plt.tight_layout()
        return plt.gcf()
        
    except Exception as e:
        print(f"Could not plot decision boundary: {e}")
        return None

def plot_learning_curves_comparison(models_data, figsize=(15, 10)):
    """
    Compare learning curves of multiple models.
    
    Args:
        models_data: List of tuples (model_name, train_scores, val_scores, train_sizes)
        figsize: Figure size
    """
    
    n_models = len(models_data)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (model_name, train_scores, val_scores, train_sizes) in enumerate(models_data):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                       alpha=0.2, color='blue')
        
        ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                       alpha=0.2, color='red')
        
        ax.set_title(model_name)
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(n_models, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        fig.delaxes(axes[row, col])
    
    plt.tight_layout()
    return fig

def create_interactive_scatter_plot(X, y, feature_names=None, title="Interactive Data Visualization"):
    """Create interactive scatter plot using Plotly."""
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    # Use first two features for scatter plot
    df = pd.DataFrame({
        'x': X[:, 0],
        'y': X[:, 1] if X.shape[1] > 1 else np.zeros(len(X)),
        'target': y
    })
    
    fig = px.scatter(df, x='x', y='y', color='target', 
                    title=title,
                    labels={'x': feature_names[0], 
                           'y': feature_names[1] if X.shape[1] > 1 else 'Constant'})
    
    fig.update_layout(
        xaxis_title=feature_names[0],
        yaxis_title=feature_names[1] if X.shape[1] > 1 else 'Constant',
        showlegend=True
    )
    
    return fig

def plot_model_complexity_curve(param_name, param_range, train_scores, val_scores, 
                               figsize=(10, 6), title="Model Complexity Curve"):
    """Plot model complexity curve (validation curve)."""
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=figsize)
    plt.plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, 
                    alpha=0.2, color='blue')
    
    plt.plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, 
                    alpha=0.2, color='red')
    
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def save_all_plots(figures, output_dir="plots", prefix=""):
    """Save all matplotlib figures to directory."""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    for i, fig in enumerate(figures):
        if fig is not None:
            filename = f"{prefix}_plot_{i+1}.png" if prefix else f"plot_{i+1}.png"
            filepath = os.path.join(output_dir, filename)
            fig.savefig(filepath, dpi=300, bbox_inches='tight')
            saved_files.append(filepath)
            
    return saved_files