"""
Machine Learning Model Training
Author: MAHBUB
Date: December 26, 2025
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve)
import xgboost as xgb
import lightgbm as lgb

sys.path.append(str(Path(__file__).parent))
from config import PROCESSED_DATA_PATHS, MODELS_DIR, RESULTS_DIR, RANDOM_STATE

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

def load_and_prepare_data():
    """Load cleaned data and prepare for modeling."""
    print("="*80)
    print("LOADING AND PREPARING DATA")
    print("="*80)
    
    df = pd.read_csv(PROCESSED_DATA_PATHS['cleaned_data'])
    print(f"\n✓ Loaded {len(df):,} records × {df.shape[1]} features")
    
    # Separate features and target
    target = 'is_ev'
    
    # Drop non-predictive columns
    drop_cols = [
        'sampno',  # ID variable
        'is_ev',  # Target
        'fuel_type_nrel',  # Derivative of target
        'fuel_category',  # Derivative of target
        'veh_class_nrel',  # Too many categories
        'region',  # Too many categories
        'county',  # Too many categories
        'comments',  # Text data
    ]
    
    # Keep only columns that exist
    drop_cols = [col for col in drop_cols if col in df.columns]
    
    X = df.drop(columns=drop_cols)
    y = df[target]
    
    print(f"\n✓ Target variable: {target}")
    print(f"  • Class 0 (Non-EV): {(y==0).sum():,} ({(y==0).mean()*100:.1f}%)")
    print(f"  • Class 1 (EV):     {(y==1).sum():,} ({(y==1).mean()*100:.1f}%)")
    print(f"\n✓ Features for modeling: {X.shape[1]}")
    
    # Handle categorical variables
    print("\n✓ Encoding categorical variables...")
    categorical_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()
    
    if len(categorical_cols) > 0:
        print(f"  • Found {len(categorical_cols)} categorical columns")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        print(f"  • After encoding: {X.shape[1]} features")
    
    return X, y

def create_train_test_split(X, y):
    """Create stratified train-test split."""
    print("\n" + "="*80)
    print("CREATING TRAIN-TEST SPLIT")
    print("="*80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.30, 
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"\n✓ Training set:   {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  • Non-EV: {(y_train==0).sum():,} ({(y_train==0).mean()*100:.1f}%)")
    print(f"  • EV:     {(y_train==1).sum():,} ({(y_train==1).mean()*100:.1f}%)")
    
    print(f"\n✓ Test set:       {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"  • Non-EV: {(y_test==0).sum():,} ({(y_test==0).mean()*100:.1f}%)")
    print(f"  • EV:     {(y_test==1).sum():,} ({(y_test==1).mean()*100:.1f}%)")
    
    # Save train/test data
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_data.to_csv(PROCESSED_DATA_PATHS['train_data'], index=False)
    test_data.to_csv(PROCESSED_DATA_PATHS['test_data'], index=False)
    print(f"\n✓ Saved train/test data to: data/processed/")
    
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train, X_test, y_test):
    """Train Logistic Regression (baseline model)."""
    print("\n" + "="*80)
    print("MODEL 1: LOGISTIC REGRESSION (Baseline)")
    print("="*80)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate class weights
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}
    
    print(f"\n✓ Class weights: {class_weight_dict}")
    
    # Train model
    print("\n✓ Training Logistic Regression...")
    lr_model = LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    lr_model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = lr_model.predict(X_test_scaled)
    y_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    # Metrics
    metrics = calculate_metrics(y_test, y_pred, y_proba, "Logistic Regression")
    
    # Save model and scaler
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr_model, MODELS_DIR / 'logistic_regression.pkl')
    joblib.dump(scaler, MODELS_DIR / 'scaler.pkl')
    print(f"\n✓ Saved model to: {MODELS_DIR / 'logistic_regression.pkl'}")
    
    return lr_model, scaler, metrics

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest."""
    print("\n" + "="*80)
    print("MODEL 2: RANDOM FOREST")
    print("="*80)
    
    print("\n✓ Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0
    )
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = calculate_metrics(y_test, y_pred, y_proba, "Random Forest")
    
    # Save model
    joblib.dump(rf_model, MODELS_DIR / 'random_forest.pkl')
    print(f"\n✓ Saved model to: {MODELS_DIR / 'random_forest.pkl'}")
    
    return rf_model, metrics

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost (primary model for SHAP)."""
    print("\n" + "="*80)
    print("MODEL 3: XGBOOST (Primary Model)")
    print("="*80)
    
    # Calculate scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\n✓ Scale_pos_weight: {scale_pos_weight:.2f}")
    
    print("\n✓ Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, verbose=False)
    
    # Predictions
    y_pred = xgb_model.predict(X_test)
    y_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = calculate_metrics(y_test, y_pred, y_proba, "XGBoost")
    
    # Save model
    joblib.dump(xgb_model, MODELS_DIR / 'xgboost.pkl')
    print(f"\n✓ Saved model to: {MODELS_DIR / 'xgboost.pkl'}")
    
    return xgb_model, metrics

def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM."""
    print("\n" + "="*80)
    print("MODEL 4: LIGHTGBM")
    print("="*80)
    
    # Clean feature names for LightGBM
    print("\n✓ Cleaning feature names for LightGBM compatibility...")
    X_train_clean = clean_feature_names(X_train)
    X_test_clean = clean_feature_names(X_test)
    
    print("✓ Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train_clean, y_train)
    
    # Predictions
    y_pred = lgb_model.predict(X_test_clean)
    y_proba = lgb_model.predict_proba(X_test_clean)[:, 1]
    
    # Metrics
    metrics = calculate_metrics(y_test, y_pred, y_proba, "LightGBM")
    
    # Save model
    joblib.dump(lgb_model, MODELS_DIR / 'lightgbm.pkl')
    print(f"\n✓ Saved model to: {MODELS_DIR / 'lightgbm.pkl'}")
    
    return lgb_model, metrics

def calculate_metrics(y_test, y_pred, y_proba, model_name):
    """Calculate and display all metrics."""
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_proba)
    }
    
    print(f"\n{'='*60}")
    print(f"PERFORMANCE METRICS: {model_name}")
    print(f"{'='*60}")
    print(f"  Accuracy:  {metrics['Accuracy']:.4f} ({metrics['Accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall:    {metrics['Recall']:.4f}")
    print(f"  F1-Score:  {metrics['F1-Score']:.4f}")
    print(f"  ROC-AUC:   {metrics['ROC-AUC']:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted")
    print(f"                Non-EV    EV")
    print(f"    Actual Non-EV  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"           EV      {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    return metrics

def compare_models(all_metrics):
    """Compare all models and save results."""
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    results_df = pd.DataFrame(all_metrics)
    
    print("\n" + results_df.to_string(index=False))
    
    # Identify best model
    best_f1 = results_df.loc[results_df['F1-Score'].idxmax()]
    best_auc = results_df.loc[results_df['ROC-AUC'].idxmax()]
    
    print(f"\n✓ Best F1-Score:  {best_f1['Model']} ({best_f1['F1-Score']:.4f})")
    print(f"✓ Best ROC-AUC:   {best_auc['Model']} ({best_auc['ROC-AUC']:.4f})")
    
    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_DIR / 'model_comparison.csv', index=False)
    print(f"\n✓ Saved comparison to: {RESULTS_DIR / 'model_comparison.csv'}")
    
    return results_df

def main():
    """Main training pipeline."""
    print("="*80)
    print("WEEK 3: MACHINE LEARNING MODEL TRAINING")
    print("="*80)
    
    # Load and prepare data
    X, y = load_and_prepare_data()
    
    # Train-test split
    X_train, X_test, y_train, y_test = create_train_test_split(X, y)
    
    # Train all models
    all_metrics = []
    
    lr_model, scaler, lr_metrics = train_logistic_regression(X_train, y_train, X_test, y_test)
    all_metrics.append(lr_metrics)
    
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    all_metrics.append(rf_metrics)
    
    xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test)
    all_metrics.append(xgb_metrics)
    
    lgb_model, lgb_metrics = train_lightgbm(X_train, y_train, X_test, y_test)
    all_metrics.append(lgb_metrics)
    
    # Compare models
    results_df = compare_models(all_metrics)
    
    print("\n" + "="*80)
    print("✓ MODEL TRAINING COMPLETE")
    print("="*80)
    print(f"\n✓ 4 models trained and saved to: {MODELS_DIR}")
    print(f"✓ Results saved to: {RESULTS_DIR / 'model_comparison.csv'}")
    print(f"\nNext step: Hyperparameter tuning (optional) or proceed to SHAP analysis")

if __name__ == "__main__":
    main()