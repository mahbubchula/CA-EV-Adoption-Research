"""
Configuration file for CA EV Research Project
Author: MAHBUB
Date: December 26, 2025
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
for dir_path in [DATA_PROCESSED, FIGURES_DIR, RESULTS_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data file paths
RAW_DATA_PATHS = {
    'residential_vehicle': DATA_RAW / "data" / "residential_vehicle.csv",
    'residential_background': DATA_RAW / "data" / "residential_background.csv",
    'residential_household': DATA_RAW / "data" / "residential_household.csv",
    'commercial_vehicle': DATA_RAW / "data" / "commercial_vehicle.csv",
    'commercial_background': DATA_RAW / "data" / "commercial_background.csv",
    'data_dictionary': DATA_RAW / "documentation" / "California_vehicle_survey_data_dictionary_2024.xlsx"
}

# Processed data paths
PROCESSED_DATA_PATHS = {
    'merged_data': DATA_PROCESSED / "merged_residential_data.csv",
    'cleaned_data': DATA_PROCESSED / "cleaned_data.csv",
    'train_data': DATA_PROCESSED / "train_data.csv",
    'test_data': DATA_PROCESSED / "test_data.csv",
    'missing_report': DATA_PROCESSED / "missing_data_report.csv"
}

# Random seed for reproducibility
RANDOM_STATE = 42

# Target variable definition
TARGET_VAR = 'is_ev'  # Binary: 1 = EV (BEV or PHEV), 0 = Gasoline

# Variables to drop (>90% missing or privacy-redacted)
VARS_TO_DROP = [
    'veh_year',  # 99.99% missing
    'veh_make',  # 99.99% missing
    'veh_model',  # 99.99% missing
    'business_miles',  # 99.73% missing
    'acquired_year_alt',  # 99.73% missing
    'persol_miles',  # 99.73% missing
    'tnc_miles',  # 96.74% missing
    'delivery_miles',  # 96.14% missing
    'electric_percent',  # 95.44% missing
]

# Figure settings (for publication quality)
FIGURE_SETTINGS = {
    'dpi': 600,
    'font_family': 'Times New Roman',
    'font_size': 12,
    'figure_format': 'png',
    'bbox_inches': 'tight'
}

# Color palettes (colorblind-friendly)
COLOR_PALETTE = {
    'primary': '#0173B2',  # Blue
    'secondary': '#DE8F05',  # Orange
    'tertiary': '#029E73',  # Green
    'quaternary': '#CC78BC',  # Purple
    'ev': '#029E73',  # Green for EV
    'non_ev': '#D55E00',  # Red-orange for Non-EV
}

# Income labels (from data dictionary)
INCOME_LABELS = {
    1: "< $10k",
    2: "$10k-$20k",
    3: "$20k-$35k",
    4: "$35k-$50k",
    5: "$50k-$75k",
    6: "$75k-$100k",
    7: "$100k-$125k",
    8: "$125k-$150k",
    9: "$150k-$200k",
    10: "$200k-$250k",
    11: "> $250k"
}

# Education labels
EDUCATION_LABELS = {
    1: "Less than high school",
    2: "High school graduate",
    3: "Some college",
    4: "Associate degree",
    5: "Bachelor's degree",
    6: "Some graduate school",
    7: "Master's degree",
    8: "Doctorate degree"
}

# Age labels
AGE_LABELS = {
    1: "16-17",
    2: "18-24",
    3: "25-34",
    4: "35-44",
    5: "45-54",
    6: "55-64",
    7: "65-74",
    8: "75+"
}

# Model hyperparameters (will be updated after tuning)
MODEL_PARAMS = {
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    },
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    },
    'lightgbm': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'random_state': RANDOM_STATE
    }
}

# Groq API settings (for LLM)
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')  # Set as environment variable
GROQ_MODEL = "llama-3.3-70b-versatile"

print("✓ Configuration loaded successfully")