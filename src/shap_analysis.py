"""
SHAP Analysis - Explainable AI
Author: MAHBUB
Date: December 26, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))
from config import (PROCESSED_DATA_PATHS, MODELS_DIR, FIGURES_DIR, 
                   FIGURE_SETTINGS, COLOR_PALETTE, RESULTS_DIR)

# Set publication-quality defaults
plt.rcParams['font.family'] = FIGURE_SETTINGS['font_family']
plt.rcParams['font.size'] = FIGURE_SETTINGS['font_size']
plt.rcParams['figure.dpi'] = FIGURE_SETTINGS['dpi']

def load_model_and_data():
    """Load XGBoost model and test data."""
    print("="*80)
    print("LOADING MODEL AND DATA")
    print("="*80)
    
    # Load model
    print("\nLoading XGBoost model...")
    model = joblib.load(MODELS_DIR / 'xgboost.pkl')
    print("  ✓ Model loaded")
    
    # Load test data
    print("\nLoading test data...")
    test_data = pd.read_csv(PROCESSED_DATA_PATHS['test_data'])
    X_test = test_data.drop(columns=['is_ev'])
    y_test = test_data['is_ev']
    print(f"  ✓ Test set: {len(X_test):,} samples × {X_test.shape[1]} features")
    
    return model, X_test, y_test

def calculate_shap_values(model, X_test, sample_size=500):
    """
    Calculate SHAP values for test set.
    Using TreeExplainer for XGBoost (fast and exact).
    """
    print("\n" + "="*80)
    print("CALCULATING SHAP VALUES")
    print("="*80)
    
    # Use a sample for faster computation
    if len(X_test) > sample_size:
        print(f"\nUsing random sample of {sample_size} observations for SHAP analysis...")
        np.random.seed(42)
        sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test.iloc[sample_indices]
    else:
        X_sample = X_test
        print(f"\nUsing all {len(X_test)} observations...")
    
    # Create SHAP explainer
    print("\nCreating SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values
    print("Calculating SHAP values (this may take 1-2 minutes)...")
    shap_values = explainer.shap_values(X_sample)
    
    print(f"  ✓ SHAP values calculated")
    print(f"  ✓ Shape: {shap_values.shape}")
    
    return explainer, shap_values, X_sample

def save_figure(fig, filename):
    """Save figure with publication settings."""
    filepath = FIGURES_DIR / filename
    fig.savefig(filepath, dpi=FIGURE_SETTINGS['dpi'], 
               bbox_inches='tight',
               format=FIGURE_SETTINGS['figure_format'])
    plt.close(fig)
    print(f"  ✓ Saved: {filename}")

def figure_11_feature_importance_bar(shap_values, X_sample):
    """
    Figure 11: Top 20 Feature Importance (Bar Chart)
    """
    print("\nCreating Figure 11: Feature Importance (Bar Chart)...")
    
    # Calculate mean absolute SHAP values
    feature_importance = pd.DataFrame({
        'feature': X_sample.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False).head(20)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = [COLOR_PALETTE['ev'] if i < 5 else COLOR_PALETTE['primary'] 
              for i in range(len(feature_importance))]
    
    ax.barh(range(len(feature_importance)), feature_importance['importance'], 
           color=colors)
    ax.set_yticks(range(len(feature_importance)))
    ax.set_yticklabels(feature_importance['feature'], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP Value| (Impact on Prediction)', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Most Important Features\n(Global Feature Importance)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    
    # Add values on bars
    for i, v in enumerate(feature_importance['importance']):
        ax.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    save_figure(fig, 'figure_11_shap_feature_importance.png')
    
    return feature_importance

def figure_12_shap_summary_plot(shap_values, X_sample):
    """
    Figure 12: SHAP Summary Plot (Beeswarm)
    Shows feature importance AND impact direction
    """
    print("\nCreating Figure 12: SHAP Summary Plot (Beeswarm)...")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create summary plot (top 20 features)
    shap.summary_plot(shap_values, X_sample, 
                     max_display=20,
                     show=False,
                     plot_size=(10, 10))
    
    plt.title('SHAP Summary Plot\nFeature Impact on EV Adoption Prediction', 
             fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('SHAP Value (Impact on Model Output)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, 'figure_12_shap_summary_beeswarm.png')

def figure_13_shap_bar_plot(shap_values, X_sample):
    """
    Figure 13: SHAP Bar Plot (Alternative View)
    """
    print("\nCreating Figure 13: SHAP Bar Plot...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    shap.plots.bar(shap.Explanation(values=shap_values, 
                                    data=X_sample.values,
                                    feature_names=X_sample.columns),
                  max_display=20,
                  show=False)
    
    plt.title('Feature Importance (Bar Plot)\nMean Absolute SHAP Values', 
             fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    save_figure(fig, 'figure_13_shap_bar_plot.png')

def figure_14_dependence_plots(shap_values, X_sample, feature_importance):
    """
    Figure 14: SHAP Dependence Plots (4-panel)
    Shows how top features affect predictions
    """
    print("\nCreating Figure 14: SHAP Dependence Plots (4-panel)...")
    
    # Get top 4 features
    top_features = feature_importance['feature'].head(4).tolist()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(top_features):
        if feature not in X_sample.columns:
            continue
        
        feature_idx = X_sample.columns.get_loc(feature)
        
        # Create scatter plot
        axes[i].scatter(X_sample[feature], shap_values[:, feature_idx],
                       c=shap_values[:, feature_idx], 
                       cmap='RdYlGn', alpha=0.6, s=20)
        
        axes[i].set_xlabel(feature, fontsize=11, fontweight='bold')
        axes[i].set_ylabel('SHAP Value\n(Impact on Prediction)', fontsize=11, fontweight='bold')
        axes[i].set_title(f'({chr(65+i)}) {feature}', fontsize=12, fontweight='bold')
        axes[i].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        axes[i].grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(axes[i].collections[0], ax=axes[i])
        cbar.set_label('SHAP Value', fontsize=9)
    
    plt.suptitle('SHAP Dependence Plots\nTop 4 Features', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    save_figure(fig, 'figure_14_shap_dependence_plots.png')

def figure_15_waterfall_plot(explainer, shap_values, X_sample):
    """
    Figure 15: SHAP Waterfall Plot (Individual Prediction Example)
    """
    print("\nCreating Figure 15: SHAP Waterfall Plot (Example Prediction)...")
    
    # Select an interesting example (an EV owner prediction)
    # Find a correctly predicted EV owner
    test_data = pd.read_csv(PROCESSED_DATA_PATHS['test_data'])
    ev_indices = test_data[test_data['is_ev'] == 1].index
    
    # Get a sample index that exists in our sample
    sample_idx = 0  # First observation in sample
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create explanation object
    explanation = shap.Explanation(
        values=shap_values[sample_idx],
        base_values=explainer.expected_value,
        data=X_sample.iloc[sample_idx].values,
        feature_names=X_sample.columns
    )
    
    shap.plots.waterfall(explanation, max_display=15, show=False)
    
    plt.title('SHAP Waterfall Plot\nIndividual Prediction Explanation', 
             fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    save_figure(fig, 'figure_15_shap_waterfall_example.png')

def save_feature_importance_table(feature_importance):
    """Save feature importance as CSV."""
    print("\nSaving feature importance table...")
    
    filepath = RESULTS_DIR / 'shap_feature_importance.csv'
    feature_importance.to_csv(filepath, index=False)
    print(f"  ✓ Saved to: {filepath}")

def main():
    """Main SHAP analysis pipeline."""
    print("="*80)
    print("WEEK 4: SHAP ANALYSIS (EXPLAINABLE AI)")
    print("="*80)
    
    # Load model and data
    model, X_test, y_test = load_model_and_data()
    
    # Calculate SHAP values
    explainer, shap_values, X_sample = calculate_shap_values(model, X_test, sample_size=500)
    
    # Generate all figures
    print("\n" + "="*80)
    print("GENERATING SHAP FIGURES (600 DPI)")
    print("="*80)
    
    feature_importance = figure_11_feature_importance_bar(shap_values, X_sample)
    figure_12_shap_summary_plot(shap_values, X_sample)
    figure_13_shap_bar_plot(shap_values, X_sample)
    figure_14_dependence_plots(shap_values, X_sample, feature_importance)
    figure_15_waterfall_plot(explainer, shap_values, X_sample)
    
    # Save results
    save_feature_importance_table(feature_importance)
    
    print("\n" + "="*80)
    print("✓ SHAP ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nGenerated 5 SHAP figures in: {FIGURES_DIR}")
    print("\nFigures created:")
    print("  • Figure 11: Feature Importance (Bar)")
    print("  • Figure 12: SHAP Summary (Beeswarm)")
    print("  • Figure 13: SHAP Bar Plot")
    print("  • Figure 14: Dependence Plots (4-panel)")
    print("  • Figure 15: Waterfall Plot (Individual)")
    print(f"\n✓ Feature importance saved to: {RESULTS_DIR / 'shap_feature_importance.csv'}")
    print("\nNext step: Hyperparameter tuning (optional) or manuscript writing!")

if __name__ == "__main__":
    main()