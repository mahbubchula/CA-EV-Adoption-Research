"""
Model Performance Visualization
Author: MAHBUB
Date: December 26, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import joblib
from sklearn.metrics import roc_curve, auc, precision_recall_curve

sys.path.append(str(Path(__file__).parent))
from config import (PROCESSED_DATA_PATHS, MODELS_DIR, FIGURES_DIR, 
                   FIGURE_SETTINGS, COLOR_PALETTE, RESULTS_DIR)

# Set publication-quality defaults
plt.rcParams['font.family'] = FIGURE_SETTINGS['font_family']
plt.rcParams['font.size'] = FIGURE_SETTINGS['font_size']
plt.rcParams['figure.dpi'] = FIGURE_SETTINGS['dpi']
sns.set_style("whitegrid")

def load_test_data():
    """Load test data for predictions."""
    print("Loading test data...")
    test_data = pd.read_csv(PROCESSED_DATA_PATHS['test_data'])
    
    # Separate features and target
    y_test = test_data['is_ev']
    X_test = test_data.drop(columns=['is_ev'])
    
    print(f"  ✓ Test set: {len(X_test):,} samples")
    return X_test, y_test

def clean_feature_names(X):
    """Clean feature names for LightGBM compatibility."""
    X = X.copy()
    X.columns = [
        col.replace('[', '_').replace(']', '_')
           .replace('<', '_').replace('>', '_')
           .replace('{', '_').replace('}', '_')
           .replace('"', '_').replace("'", '_')
           .replace(':', '_').replace(',', '_')
           .replace(' ', '_')
        for col in X.columns
    ]
    return X

def get_model_predictions():
    """Get predictions from all models."""
    print("\nGenerating predictions from all models...")
    
    X_test, y_test = load_test_data()
    
    predictions = {}
    
    # Logistic Regression (needs scaling)
    print("  • Logistic Regression...")
    scaler = joblib.load(MODELS_DIR / 'scaler.pkl')
    lr_model = joblib.load(MODELS_DIR / 'logistic_regression.pkl')
    X_test_scaled = scaler.transform(X_test)
    predictions['Logistic Regression'] = {
        'y_pred': lr_model.predict(X_test_scaled),
        'y_proba': lr_model.predict_proba(X_test_scaled)[:, 1]
    }
    
    # Random Forest
    print("  • Random Forest...")
    rf_model = joblib.load(MODELS_DIR / 'random_forest.pkl')
    predictions['Random Forest'] = {
        'y_pred': rf_model.predict(X_test),
        'y_proba': rf_model.predict_proba(X_test)[:, 1]
    }
    
    # XGBoost
    print("  • XGBoost...")
    xgb_model = joblib.load(MODELS_DIR / 'xgboost.pkl')
    predictions['XGBoost'] = {
        'y_pred': xgb_model.predict(X_test),
        'y_proba': xgb_model.predict_proba(X_test)[:, 1]
    }
    
    # LightGBM
    print("  • LightGBM...")
    lgb_model = joblib.load(MODELS_DIR / 'lightgbm.pkl')
    X_test_clean = clean_feature_names(X_test)
    predictions['LightGBM'] = {
        'y_pred': lgb_model.predict(X_test_clean),
        'y_proba': lgb_model.predict_proba(X_test_clean)[:, 1]
    }
    
    return predictions, y_test

def save_figure(fig, filename):
    """Save figure with publication settings."""
    filepath = FIGURES_DIR / filename
    fig.savefig(filepath, dpi=FIGURE_SETTINGS['dpi'], 
               bbox_inches=FIGURE_SETTINGS['bbox_inches'],
               format=FIGURE_SETTINGS['figure_format'])
    plt.close(fig)
    print(f"  ✓ Saved: {filename}")

def figure_6_model_comparison_bars(comparison_df):
    """
    Figure 6: Comprehensive Model Comparison (6-panel)
    """
    print("\nCreating Figure 6: Model Comparison (6-panel)...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    models = comparison_df['Model'].tolist()
    colors = [COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], 
              COLOR_PALETTE['ev'], COLOR_PALETTE['quaternary']]
    
    # Panel A: Accuracy
    axes[0, 0].bar(models, comparison_df['Accuracy'], color=colors)
    axes[0, 0].set_title('(A) Accuracy', fontweight='bold', fontsize=14)
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].set_ylim(0.75, 1.0)
    axes[0, 0].axhline(y=0.95, color='red', linestyle='--', linewidth=1, alpha=0.5)
    for i, v in enumerate(comparison_df['Accuracy']):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
    plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Panel B: Precision
    axes[0, 1].bar(models, comparison_df['Precision'], color=colors)
    axes[0, 1].set_title('(B) Precision', fontweight='bold', fontsize=14)
    axes[0, 1].set_ylabel('Precision', fontsize=12)
    axes[0, 1].set_ylim(0, 1.0)
    for i, v in enumerate(comparison_df['Precision']):
        axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Panel C: Recall
    axes[0, 2].bar(models, comparison_df['Recall'], color=colors)
    axes[0, 2].set_title('(C) Recall (Sensitivity)', fontweight='bold', fontsize=14)
    axes[0, 2].set_ylabel('Recall', fontsize=12)
    axes[0, 2].set_ylim(0, 1.0)
    for i, v in enumerate(comparison_df['Recall']):
        axes[0, 2].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    plt.setp(axes[0, 2].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Panel D: F1-Score
    axes[1, 0].bar(models, comparison_df['F1-Score'], color=colors)
    axes[1, 0].set_title('(D) F1-Score (Balanced)', fontweight='bold', fontsize=14)
    axes[1, 0].set_ylabel('F1-Score', fontsize=12)
    axes[1, 0].set_ylim(0, 1.0)
    axes[1, 0].axhline(y=0.8, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Excellent (>0.8)')
    axes[1, 0].legend()
    for i, v in enumerate(comparison_df['F1-Score']):
        axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Panel E: ROC-AUC
    axes[1, 1].bar(models, comparison_df['ROC-AUC'], color=colors)
    axes[1, 1].set_title('(E) ROC-AUC', fontweight='bold', fontsize=14)
    axes[1, 1].set_ylabel('ROC-AUC', fontsize=12)
    axes[1, 1].set_ylim(0.8, 1.0)
    axes[1, 1].axhline(y=0.95, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Excellent (>0.95)')
    axes[1, 1].legend()
    for i, v in enumerate(comparison_df['ROC-AUC']):
        axes[1, 1].text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Panel F: Overall Performance Radar
    # Create spider/radar chart showing all metrics
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    # Best two models only (XGBoost and LightGBM)
    xgb_values = comparison_df[comparison_df['Model'] == 'XGBoost'][categories].values[0]
    lgb_values = comparison_df[comparison_df['Model'] == 'LightGBM'][categories].values[0]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    xgb_values = np.concatenate((xgb_values, [xgb_values[0]]))
    lgb_values = np.concatenate((lgb_values, [lgb_values[0]]))
    angles += angles[:1]
    
    ax = plt.subplot(2, 3, 6, projection='polar')
    ax.plot(angles, xgb_values, 'o-', linewidth=2, label='XGBoost', color=COLOR_PALETTE['ev'])
    ax.fill(angles, xgb_values, alpha=0.25, color=COLOR_PALETTE['ev'])
    ax.plot(angles, lgb_values, 's-', linewidth=2, label='LightGBM', color=COLOR_PALETTE['quaternary'])
    ax.fill(angles, lgb_values, alpha=0.25, color=COLOR_PALETTE['quaternary'])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0.8, 1.0)
    ax.set_title('(F) Top Models Comparison\n(Radar Chart)', fontweight='bold', fontsize=14, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    save_figure(fig, 'figure_06_model_comparison_comprehensive.png')

def figure_7_roc_curves(predictions, y_test):
    """
    Figure 7: ROC Curves for All Models
    """
    print("\nCreating Figure 7: ROC Curves...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = [COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], 
              COLOR_PALETTE['ev'], COLOR_PALETTE['quaternary']]
    
    for i, (model_name, pred_data) in enumerate(predictions.items()):
        fpr, tpr, _ = roc_curve(y_test, pred_data['y_proba'])
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color=colors[i], lw=3, 
               label=f'{model_name} (AUC = {roc_auc:.4f})')
    
    # Diagonal line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.5000)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=14, fontweight='bold')
    ax.set_title('Receiver Operating Characteristic (ROC) Curves\nAll Models Comparison', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add annotation for top-right corner (perfect classifier region)
    ax.annotate('Perfect Classifier\n(100% TPR, 0% FPR)', 
               xy=(0, 1), xytext=(0.3, 0.85),
               arrowprops=dict(arrowstyle='->', color='green', lw=2),
               fontsize=10, color='green', fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, 'figure_07_roc_curves.png')

def figure_8_precision_recall_curves(predictions, y_test):
    """
    Figure 8: Precision-Recall Curves
    """
    print("\nCreating Figure 8: Precision-Recall Curves...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = [COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], 
              COLOR_PALETTE['ev'], COLOR_PALETTE['quaternary']]
    
    # Calculate baseline (random classifier)
    baseline = (y_test == 1).sum() / len(y_test)
    
    for i, (model_name, pred_data) in enumerate(predictions.items()):
        precision, recall, _ = precision_recall_curve(y_test, pred_data['y_proba'])
        
        # Calculate average precision
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_test, pred_data['y_proba'])
        
        ax.plot(recall, precision, color=colors[i], lw=3, 
               label=f'{model_name} (AP = {ap:.4f})')
    
    # Baseline (random classifier)
    ax.axhline(y=baseline, color='k', linestyle='--', lw=2, 
              label=f'Random Classifier (AP = {baseline:.4f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (True Positive Rate)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=14, fontweight='bold')
    ax.set_title('Precision-Recall Curves\nAll Models Comparison', 
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='lower left', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'figure_08_precision_recall_curves.png')

def figure_9_confusion_matrices(predictions, y_test):
    """
    Figure 9: Confusion Matrices (2x2 grid)
    """
    print("\nCreating Figure 9: Confusion Matrices...")
    
    from sklearn.metrics import confusion_matrix
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    
    model_names = list(predictions.keys())
    
    for i, model_name in enumerate(model_names):
        cm = confusion_matrix(y_test, predictions[model_name]['y_pred'])
        
        # Normalize by row (true labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   ax=axes[i], cbar=False,
                   xticklabels=['Non-EV', 'EV'],
                   yticklabels=['Non-EV', 'EV'],
                   annot_kws={'fontsize': 14, 'fontweight': 'bold'})
        
        # Add percentages
        for j in range(2):
            for k in range(2):
                axes[i].text(k + 0.5, j + 0.7, f'({cm_normalized[j, k]*100:.1f}%)',
                           ha='center', va='center', fontsize=10, color='gray')
        
        axes[i].set_title(f'({chr(65+i)}) {model_name}', 
                         fontweight='bold', fontsize=14, pad=10)
        axes[i].set_ylabel('Actual Class', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
        
        # Add accuracy in corner
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        axes[i].text(0.98, 0.02, f'Accuracy: {accuracy:.3f}',
                    transform=axes[i].transAxes,
                    ha='right', va='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, 'figure_09_confusion_matrices.png')

def figure_10_performance_heatmap(comparison_df):
    """
    Figure 10: Performance Heatmap (All Metrics)
    """
    print("\nCreating Figure 10: Performance Heatmap...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    metrics_data = comparison_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]
    
    # Create heatmap
    sns.heatmap(metrics_data.T, annot=True, fmt='.4f', cmap='RdYlGn', 
               vmin=0.5, vmax=1.0, center=0.85,
               linewidths=2, linecolor='white',
               cbar_kws={'label': 'Performance Score'},
               ax=ax, annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    
    ax.set_title('Model Performance Heatmap\n(All Metrics)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=14, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    save_figure(fig, 'figure_10_performance_heatmap.png')

def main():
    """Main function to generate all model performance figures."""
    print("="*80)
    print("MODEL PERFORMANCE VISUALIZATION")
    print("="*80)
    
    # Load comparison data
    comparison_df = pd.read_csv(RESULTS_DIR / 'model_comparison.csv')
    
    # Get predictions
    predictions, y_test = get_model_predictions()
    
    # Generate all figures
    print("\n" + "="*80)
    print("GENERATING PUBLICATION-GRADE FIGURES (600 DPI)")
    print("="*80)
    
    figure_6_model_comparison_bars(comparison_df)
    figure_7_roc_curves(predictions, y_test)
    figure_8_precision_recall_curves(predictions, y_test)
    figure_9_confusion_matrices(predictions, y_test)
    figure_10_performance_heatmap(comparison_df)
    
    print("\n" + "="*80)
    print("✓ ALL MODEL PERFORMANCE FIGURES GENERATED")
    print("="*80)
    print(f"\nGenerated 5 figures in: {FIGURES_DIR}")
    print("\nFigures created:")
    print("  • Figure 6: Model Comparison (6-panel)")
    print("  • Figure 7: ROC Curves")
    print("  • Figure 8: Precision-Recall Curves")
    print("  • Figure 9: Confusion Matrices (4 models)")
    print("  • Figure 10: Performance Heatmap")
    print("\nNext step: Week 4 - SHAP Analysis (Explainable AI)")

if __name__ == "__main__":
    main()