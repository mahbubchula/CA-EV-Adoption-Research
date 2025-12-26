# California EV Adoption Research Project
## Interpretable Machine Learning for Electric Vehicle Adoption

**Author**: MAHBUB (Chulalongkorn University)  
**Date**: December 26, 2025  
**Target**: Q2 Journal Publication (Transport Policy / Energy Research & Social Science)

---

## 📁 PROJECT STRUCTURE

```
CA_EV_Research/
├── data/
│   ├── raw/                    # Original NREL dataset (DO NOT MODIFY)
│   │   ├── data/
│   │   │   ├── residential_vehicle.csv
│   │   │   ├── residential_background.csv
│   │   │   ├── residential_household.csv
│   │   │   ├── commercial_vehicle.csv
│   │   │   └── commercial_background.csv
│   │   └── documentation/
│   │       └── California_vehicle_survey_data_dictionary_2024.xlsx
│   └── processed/              # Processed data (auto-generated)
│       ├── merged_residential_data.csv
│       ├── cleaned_data.csv
│       ├── train_data.csv
│       ├── test_data.csv
│       └── missing_data_report.csv
├── src/                        # Source code
│   ├── config.py              # Configuration & settings
│   ├── data_loader.py         # Data loading & merging
│   ├── data_cleaning.py       # Missing data handling
│   ├── feature_engineering.py # Create derived features
│   ├── exploratory_analysis.py # EDA & visualizations
│   ├── modeling.py            # ML model training
│   ├── shap_analysis.py       # XAI analysis
│   └── llm_integration.py     # LLM-powered insights
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_results_analysis.ipynb
├── figures/                    # Publication-grade figures (600 DPI)
├── results/                    # Model outputs, SHAP values
├── models/                     # Trained models (.pkl files)
├── streamlit_app/             # Interactive web app
│   └── app.py
├── docs/                       # Documentation & reports
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 GETTING STARTED

### **Prerequisites**
- Python 3.9 or higher
- VS Code (recommended) or any Python IDE
- Git (optional, for version control)

### **Step 1: Install Python Packages**

Open terminal in project root directory:

```bash
# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### **Step 2: Set Up Data**

1. **Extract dataset** to `data/raw/`
2. **Verify file structure**:
   ```bash
   ls data/raw/data/
   # Should show: residential_vehicle.csv, residential_background.csv, etc.
   ```

### **Step 3: Configure Environment**

**For LLM features (optional):**
```bash
# Set Groq API key as environment variable
# On Windows (PowerShell):
$env:GROQ_API_KEY = "your_api_key_here"

# On Mac/Linux:
export GROQ_API_KEY="your_api_key_here"

# Get free API key at: https://console.groq.com/
```

---

## 📊 WEEK 1: DATA PREPARATION (CURRENT)

### **Step 1: Data Merging**

```bash
cd src
python data_loader.py
```

**What it does:**
- Loads all 3 residential datasets
- Merges vehicles + background + household data
- Creates binary target variable (`is_ev`)
- Saves to: `data/processed/merged_residential_data.csv`

**Expected output:**
```
✓ Loaded 7,353 vehicles from 3,800 households
✓ Target: 888 EV (12.1%), 6,063 Non-EV (87.9%)
✓ Merged dataset saved
```

### **Step 2: Data Cleaning**

```bash
python data_cleaning.py
```

**What it does:**
- Analyzes missing data patterns
- Drops variables with >90% missing
- Imputes numerical variables (grouped median)
- Imputes binary EV experience (0 if missing)
- Imputes categorical variables (mode)
- Saves to: `data/processed/cleaned_data.csv`
- Creates: `data/processed/missing_data_report.csv`

**Expected output:**
```
✓ Dropped 9 variables (>90% missing)
✓ Imputed 15 numerical variables
✓ No missing data remaining
✓ Final dataset: 6,951 records × 312 features
```

### **Step 3: Feature Engineering**

```bash
python feature_engineering.py
```

**What it does:**
- Creates 10+ new derived features:
  1. EV Experience Score (0-3 scale)
  2. Charging Access Index (0-1 scale)
  3. Income Categories (Low/Medium/High)
  4. College Degree Binary
  5. Vehicle Age Approximation
  6. Affordability Ratio (price/income)
  7. Mileage Categories (Low/Medium/High)
  8. Multi-Vehicle Household Indicator
  9. Urban Region Indicator
  10. Adoption Readiness Score (0-10 scale)

**Expected output:**
```
✓ Created 10 new features
✓ Total features: 322
```

---

## 🔍 USAGE EXAMPLES

### **Example 1: Load and Explore Data**

```python
import pandas as pd
import sys
sys.path.append('src')
from config import PROCESSED_DATA_PATHS

# Load cleaned data
df = pd.read_csv(PROCESSED_DATA_PATHS['cleaned_data'])

# Check shape
print(f"Dataset: {df.shape[0]:,} records × {df.shape[1]} features")

# Check target distribution
print(df['is_ev'].value_counts())

# Check new engineered features
print(df[['ev_experience_score', 'charging_access_index', 
          'adoption_readiness_score']].describe())
```

### **Example 2: Run Full Week 1 Pipeline**

```python
# Create a script: run_week1.py

import sys
sys.path.append('src')

from data_loader import main as load_data
from data_cleaning import main as clean_data
from feature_engineering import main as engineer_features

print("WEEK 1: DATA PREPARATION PIPELINE")
print("="*80)

# Step 1: Load and merge
print("\n[1/3] Loading and merging data...")
merged_df = load_data()

# Step 2: Clean data
print("\n[2/3] Cleaning data...")
cleaned_df = clean_data()

# Step 3: Engineer features
print("\n[3/3] Engineering features...")
final_df = engineer_features()

print("\n✓ WEEK 1 COMPLETE!")
print(f"Final dataset: {final_df.shape[0]:,} records × {final_df.shape[1]} features")
```

Then run:
```bash
python run_week1.py
```

---

## 📈 NEXT STEPS (Week 2+)

### **Week 2: Exploratory Data Analysis**
- `python exploratory_analysis.py`
- Generates 15+ publication-grade figures (600 DPI)
- Creates descriptive statistics table
- Performs statistical tests

### **Week 3: Machine Learning Modeling**
- `python modeling.py`
- Trains 4 models (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Hyperparameter tuning with Optuna
- Saves trained models to `models/`

### **Week 4-5: SHAP Analysis**
- `python shap_analysis.py`
- Global feature importance
- Dependence plots
- Individual prediction explanations
- Subgroup analysis (equity)

### **Week 6: Streamlit App**
- `streamlit run streamlit_app/app.py`
- Interactive EV adoption predictor
- Policy scenario simulator
- Equity dashboard

---

## 📊 DATA DICTIONARY

### **Target Variable**
- `is_ev`: Binary (1 = EV, 0 = Gasoline)
- `fuel_category`: Multi-class (0 = Gasoline, 1 = PHEV, 2 = BEV)

### **Key Features**

**Demographics:**
- `income`: 1-11 (< $10k to > $250k)
- `max_age`: 1-8 (16-17 to 75+)
- `max_education`: 1-8 (Less than HS to Doctorate)
- `household_size`: Number of household members

**EV Experience:**
- `hybrid_experience`: Binary (1 = Yes, 0 = No)
- `phev_experience`: Binary
- `bev_experience`: Binary
- `ev_experience_score`: 0-3 (composite score)

**Charging Infrastructure:**
- `home_charge`: Binary or Yes/No
- `charge_work`: Binary or Yes/No
- `charge_spots`: Number of public charging locations aware of
- `charging_access_index`: 0-1 (composite index)

**Vehicle Characteristics:**
- `annual_mileage`: Miles per year
- `veh_class_nrel`: Vehicle class (SUV, sedan, etc.)
- `acquired_price_1`: Purchase price (dollars)
- `vehicle_age_approx`: Years since purchase

**Geography:**
- `county`: California county
- `region`: Geographic region
- `urban_region`: Binary (urban vs. rural)

**Engineered Features:**
- `income_category`: Low/Medium/High
- `college_degree_plus`: Binary
- `affordability_ratio`: Price/income
- `mileage_category`: Low/Medium/High
- `multi_vehicle_household`: Binary
- `adoption_readiness_score`: 0-10 composite metric

---

## ⚙️ CONFIGURATION

Edit `src/config.py` to customize:

```python
# Random seed (for reproducibility)
RANDOM_STATE = 42

# Figure settings
FIGURE_SETTINGS = {
    'dpi': 600,  # Publication quality
    'font_family': 'Times New Roman',
    'font_size': 12
}

# Model hyperparameters
MODEL_PARAMS = {
    'xgboost': {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        # ... etc
    }
}
```

---

## 🐛 TROUBLESHOOTING

### **Problem: "File not found" error**
**Solution**: Check that data is in `data/raw/data/` directory
```bash
ls data/raw/data/
# Should show all CSV files
```

### **Problem: "Module not found" error**
**Solution**: Make sure you're in the project root and virtual environment is activated
```bash
# Activate venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate  # Windows

# Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # Mac/Linux
$env:PYTHONPATH += ";$(Get-Location)\src"  # Windows
```

### **Problem: "Memory error" when loading data**
**Solution**: Process in chunks or use smaller sample
```python
# In data_loader.py, modify:
df = pd.read_csv(path, low_memory=False, nrows=5000)  # Sample first 5000 rows
```

### **Problem: Figures not displaying in VS Code**
**Solution**: Install Python extension and use Jupyter notebooks
```bash
# Convert script to notebook
jupyter nbconvert --to notebook src/exploratory_analysis.py
```

---

## 📝 DEVELOPMENT WORKFLOW

### **Recommended VS Code Extensions:**
1. **Python** (Microsoft) - IntelliSense, debugging
2. **Jupyter** (Microsoft) - Notebook support
3. **Pylance** (Microsoft) - Type checking
4. **autoDocstring** - Generate docstrings
5. **GitLens** (optional) - Git integration

### **Daily Workflow:**
```bash
# 1. Activate environment
source venv/bin/activate

# 2. Pull latest changes (if using Git)
git pull

# 3. Run your work
cd src
python your_script.py

# 4. Check results
ls ../data/processed/
ls ../figures/

# 5. Commit changes (if using Git)
git add .
git commit -m "Completed Week 1 data preparation"
git push
```

---

## 📚 RESOURCES

### **Documentation:**
- NREL Dataset: https://www.nrel.gov/transportation/secure-transportation-data/tsdc-cleansed-data/2024-california-vehicle-survey
- XGBoost: https://xgboost.readthedocs.io/
- SHAP: https://shap.readthedocs.io/
- Streamlit: https://docs.streamlit.io/

### **Key Papers:**
- SHAP: Lundberg & Lee (2017) - "A Unified Approach to Interpreting Model Predictions"
- EV Adoption: Rezvani et al. (2015) - "Advances in consumer electric vehicle adoption research"

---

## 🎯 PROJECT GOALS

### **Week 1** ✅ (Current)
- [x] Data merging
- [x] Missing data handling
- [x] Feature engineering

### **Week 2** (Next)
- [ ] Exploratory data analysis
- [ ] Descriptive statistics
- [ ] 15+ publication figures

### **Week 3-4** (ML Modeling)
- [ ] Train 4 models
- [ ] Hyperparameter tuning
- [ ] Model evaluation

### **Week 5-6** (XAI + Tool)
- [ ] SHAP analysis
- [ ] Equity analysis
- [ ] Streamlit app

### **Week 7-10** (Manuscript)
- [ ] Write 10,000-word paper
- [ ] Create appendices
- [ ] Submit to journal

---

## 🤝 SUPPORT

**Questions?** Contact MAHBUB:
- Email: 6870376421@student.chula.ac.th
- University: Chulalongkorn University
- Department: Transportation Engineering

---

## 📄 LICENSE & CITATION

### **Data Source:**
```
Transportation Secure Data Center (2024). 
California Vehicle Survey 2024. 
National Renewable Energy Laboratory. 
www.nrel.gov/tsdc
```

### **Code License:**
MIT License - Feel free to use for research and education.

### **How to Cite (when published):**
```
[Your Name] (2025). 
Interpretable Machine Learning for Electric Vehicle Adoption: 
A California Household Analysis. 
[Journal Name], [Volume]([Issue]), [Pages].
```

---

**Last Updated**: December 26, 2025  
**Version**: 1.0 (Week 1 Complete)  
**Status**: 🟢 Data Preparation Complete, Ready for EDA