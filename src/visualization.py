"""
Visualization Module for Hospital Readmission Prediction

This module handles:
- ROC curve generation
- Precision-Recall curve generation
- Confusion matrix visualization
- Model comparison charts
- Feature importance plots
- Class distribution plots
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from typing import Optional, List, Tuple
import os


# Plot styling configuration
PLOT_STYLE = {
    'figure.figsize': (10, 8),
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold'
}

COLORS = {
    'primary': '#3498db',
    'secondary': '#e74c3c',
    'success': '#2ecc71',
    'warning': '#f39c12',
    'info': '#9b59b6'
}


def setup_plot_style():
    """Apply consistent styling to all plots."""
    plt.rcParams.update(PLOT_STYLE)
    sns.set_palette("husl")


def save_roc_curve(y_true: np.ndarray, 
                   y_prob: np.ndarray,
                   auc_score: float,
                   output_path: str,
                   title: str = "ROC Curve - Hospital Readmission Prediction"):
    """
    Generate and save ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        auc_score: AUC-ROC score
        output_path: Path to save image
        title: Plot title
    """
    setup_plot_style()
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, 'b-', linewidth=2.5, 
             label=f'Model (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, 
             label='Random Classifier (AUC = 0.50)')
    
    # Add operating point at threshold 0.5
    default_idx = np.argmin(np.abs(thresholds - 0.5))
    plt.scatter(fpr[default_idx], tpr[default_idx], s=200, c='green', 
               marker='o', zorder=5, 
               label=f'Threshold 0.5 (TPR={tpr[default_idx]:.3f}, FPR={fpr[default_idx]:.3f})')
    
    plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    plt.title(title, fontsize=15, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved ROC curve to: {output_path}")


def save_precision_recall_curve(y_true: np.ndarray,
                                y_prob: np.ndarray,
                                auc_pr_score: float,
                                output_path: str,
                                title: str = "Precision-Recall Curve - Hospital Readmission Prediction"):
    """
    Generate and save Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        auc_pr_score: AUC-PR score
        output_path: Path to save image
        title: Plot title
    """
    setup_plot_style()
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    baseline = y_true.mean()
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, 'b-', linewidth=2.5,
             label=f'Model (AUC-PR = {auc_pr_score:.4f})')
    plt.axhline(y=baseline, color='r', linestyle='--', linewidth=2,
                label=f'Baseline = {baseline:.4f}')
    
    plt.xlabel('Recall (Sensitivity)', fontsize=13, fontweight='bold')
    plt.ylabel('Precision', fontsize=13, fontweight='bold')
    plt.title(title, fontsize=15, fontweight='bold', pad=20)
    plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved Precision-Recall curve to: {output_path}")


def save_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         output_path: str,
                         labels: List[str] = ['No Readmit', 'Readmit'],
                         title: str = "Confusion Matrix - Hospital Readmission Prediction"):
    """
    Generate and save confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_path: Path to save image
        labels: Class labels
        title: Plot title
    """
    setup_plot_style()
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', 
                cbar_kws={'label': 'Count'},
                xticklabels=labels,
                yticklabels=labels,
                annot_kws={'size': 16, 'fontweight': 'bold'})
    
    plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
    plt.ylabel('Actual Label', fontsize=13, fontweight='bold')
    plt.title(title, fontsize=15, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved confusion matrix to: {output_path}")


def save_model_comparison(comparison_df: pd.DataFrame,
                         output_path: str,
                         metrics: List[str] = ['AUC_ROC', 'AUC_PR', 'F1_Score', 'Accuracy'],
                         title: str = "Model Performance Comparison"):
    """
    Generate and save model comparison chart.
    
    Args:
        comparison_df: DataFrame with model comparison metrics
        output_path: Path to save image
        metrics: List of metrics to compare
        title: Plot title
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    
    colors = [COLORS['primary'], COLORS['secondary'], 
              COLORS['success'], COLORS['warning']]
    
    for idx, (ax, metric, color) in enumerate(zip(axes.flat, metrics, colors)):
        models = comparison_df['Model'].values
        values = comparison_df[metric].values
        
        bars = ax.barh(models, values, color=color, alpha=0.7, 
                      edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            ax.text(val + 0.01, i, f'{val:.4f}', 
                   va='center', fontweight='bold', fontsize=10)
        
        ax.set_xlabel(metric.replace('_', ' '), fontsize=11, fontweight='bold')
        ax.set_xlim([0, max(values) * 1.15])
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved model comparison to: {output_path}")


def save_feature_importance(feature_names: List[str],
                           importance_values: np.ndarray,
                           output_path: str,
                           top_n: int = 15,
                           title: str = "Top Feature Importances"):
    """
    Generate and save feature importance chart.
    
    Args:
        feature_names: List of feature names
        importance_values: Array of importance scores
        output_path: Path to save image
        top_n: Number of top features to display
        title: Plot title
    """
    setup_plot_style()
    
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_values
    }).sort_values('Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(importance_df['Feature'], importance_df['Importance'],
                   color=COLORS['primary'], alpha=0.8, 
                   edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', ha='left', va='center', 
                fontweight='bold', fontsize=9)
    
    plt.xlabel('Importance Score', fontsize=13, fontweight='bold')
    plt.ylabel('Feature', fontsize=13, fontweight='bold')
    plt.title(title, fontsize=15, fontweight='bold', pad=20)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved feature importance to: {output_path}")


def save_class_distribution(y_true: np.ndarray,
                           output_path: str,
                           labels: List[str] = ['No Readmit', 'Readmit'],
                           title: str = "Target Variable Distribution"):
    """
    Generate and save class distribution chart.
    
    Args:
        y_true: True labels
        output_path: Path to save image
        labels: Class labels
        title: Plot title
    """
    setup_plot_style()
    
    class_counts = [len(y_true[y_true==0]), len(y_true[y_true==1])]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Pie chart
    colors = [COLORS['primary'], COLORS['secondary']]
    explode = (0.05, 0.05)
    ax1.pie(class_counts, labels=labels, autopct='%1.1f%%',
           colors=colors, explode=explode, shadow=True, startangle=90,
           textprops={'fontsize': 12, 'fontweight': 'bold'})
    ax1.set_title('Proportion', fontsize=13, fontweight='bold', pad=10)
    
    # Bar chart
    bars = ax2.bar(labels, class_counts, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=2)
    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}\n({count/sum(class_counts)*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax2.set_title('Absolute Counts', fontsize=13, fontweight='bold', pad=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved class distribution to: {output_path}")


def save_all_visualizations(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           y_prob: np.ndarray,
                           comparison_df: pd.DataFrame,
                           auc_roc: float,
                           auc_pr: float,
                           output_dir: str = "images",
                           feature_names: Optional[List[str]] = None,
                           feature_importance: Optional[np.ndarray] = None):
    """
    Generate and save all visualizations.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        comparison_df: Model comparison DataFrame
        auc_roc: AUC-ROC score
        auc_pr: AUC-PR score
        output_dir: Directory to save images
        feature_names: Optional feature names for importance plot
        feature_importance: Optional feature importance values
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING ALL VISUALIZATIONS")
    print("=" * 70)
    
    # ROC Curve
    save_roc_curve(y_true, y_prob, auc_roc, 
                   f"{output_dir}/roc_curve.png")
    
    # Precision-Recall Curve
    save_precision_recall_curve(y_true, y_prob, auc_pr,
                               f"{output_dir}/precision_recall_curve.png")
    
    # Confusion Matrix
    save_confusion_matrix(y_true, y_pred,
                         f"{output_dir}/confusion_matrix.png")
    
    # Model Comparison
    save_model_comparison(comparison_df,
                         f"{output_dir}/model_comparison.png")
    
    # Class Distribution
    save_class_distribution(y_true,
                           f"{output_dir}/class_distribution.png")
    
    # Feature Importance (if provided)
    if feature_names is not None and feature_importance is not None:
        save_feature_importance(feature_names, feature_importance,
                               f"{output_dir}/feature_importance.png")
    
    print("=" * 70)
    print("✅ ALL VISUALIZATIONS SAVED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nImages saved to: {output_dir}/")
