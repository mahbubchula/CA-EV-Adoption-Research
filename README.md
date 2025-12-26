# 🚗 California EV Adoption Research

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-96%25-00C853?style=for-the-badge&logo=xgboost&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

**Interpretable Machine Learning for Electric Vehicle Adoption Analysis**

[📊 Dataset](#-dataset) • [🎯 Results](#-key-results) • [🚀 Quick Start](#-quick-start) • [📱 Demo](#-interactive-dashboard) • [📄 Paper](#-publication)

</div>

---

## 📖 Overview

A comprehensive machine learning study analyzing **Electric Vehicle (EV) adoption patterns** among **7,353 California households** using state-of-the-art interpretable AI techniques.

### 🌟 Highlights

- **🎯 96% Prediction Accuracy** - XGBoost model with ROC-AUC of 0.976
- **🔍 Explainable AI** - SHAP analysis revealing key adoption drivers
- **📊 15 Publication Figures** - 600 DPI, publication-ready visualizations
- **📱 Interactive Dashboard** - 6-page Streamlit web application
- **📈 Statistical Rigor** - All tests p < 0.001, strong effect sizes

---

## 🎯 Key Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|:------|:--------:|:---------:|:------:|:--------:|:-------:|
| **XGBoost** ⭐ | **95.97%** | **0.813** | **0.865** | **0.838** | **0.976** |
| LightGBM | 95.92% | 0.803 | 0.876 | 0.838 | 0.975 |
| Random Forest | 81.10% | 0.378 | 0.884 | 0.530 | 0.909 |
| Logistic Regression | 88.98% | 0.535 | 0.665 | 0.593 | 0.884 |

### Top 5 Predictors (SHAP Analysis)

1. 💰 **Income Level** - Strongest predictor (Cohen's d = 0.48)
2. 🎓 **Education** - 94% of EV owners have college degree vs 80% non-EV
3. ⏰ **Vehicle Age** - EVs are 3 years newer on average
4. 🔌 **Charging Access** - Infrastructure availability matters
5. 📊 **Adoption Readiness** - Composite behavioral score

### Key Findings

- ✅ **Income effect**: EV owners earn ~$25k more ($125-150k vs $100-125k)
- ✅ **Education gap**: 14.2% higher college degree rate among EV owners
- ✅ **Age matters**: EV owners purchase significantly newer vehicles (4.0 vs 7.1 years)
- ✅ **All demographic variables**: Statistically significant (p < 0.001)

---

## 📊 Dataset

**Source**: NREL California Vehicle Survey 2024  
**Sample**: 7,353 vehicles from 3,800 households  
**Target**: Binary EV adoption (12.1% adoption rate)  
**Features**: 244 variables (232 original + 12 engineered)

### Engineered Features

- `ev_experience_score` - Composite hybrid/PHEV/BEV experience (0-2)
- `charging_access_index` - Home + work + public charging (0-1)
- `adoption_readiness_score` - Multi-factor readiness metric (0-10)
- `income_category` - Low/Medium/High income brackets
- `college_degree_plus` - Bachelor's degree or higher

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip package manager
- 8GB RAM minimum

### Installation
```bash
# Clone repository
git clone https://github.com/mahbubchula/CA-EV-Adoption-Research.git
cd CA-EV-Adoption-Research

# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### Data Setup

⚠️ **Dataset not included** (proprietary NREL data, too large for GitHub)

1. Download from: [NREL Transportation Secure Data Center](https://www.nrel.gov/transportation/secure-transportation-data/)
2. Extract to: `data/raw/data/`
3. Required files:
   - `residential_vehicle.csv`
   - `residential_background.csv`
   - `residential_household.csv`

### Run Pipeline
```bash
# Navigate to source directory
cd src

# Step 1: Load and merge data
python data_loader.py

# Step 2: Clean data
python data_cleaning.py

# Step 3: Engineer features
python feature_engineering.py

# Step 4: Train models
python train_models.py

# Step 5: SHAP analysis
python shap_analysis.py

# Step 6: Generate visualizations
python eda_visualizations.py
python visualize_model_results.py
```

---

## 📱 Interactive Dashboard

Launch the Streamlit web application:
```bash
streamlit run streamlit_app/app.py
```

### Dashboard Features

- 🔮 **EV Predictor** - Input household characteristics, get instant predictions
- 📊 **Model Performance** - Interactive metrics, confusion matrices, ROC curves
- 🔍 **Feature Importance** - SHAP visualizations and explanations
- 💡 **Policy Simulator** - Test what-if scenarios for policy interventions
- 📈 **Data Explorer** - Interactive filtering and visualization
- 📋 **Statistical Tests** - Chi-square, t-tests, correlation analysis

---

## 📁 Project Structure
```
CA_EV_Research/
├── 📂 src/                     # Python source code
│   ├── config.py               # Project configuration
│   ├── data_loader.py          # Data merging pipeline
│   ├── data_cleaning.py        # Missing data handling
│   ├── feature_engineering.py  # Feature creation
│   ├── train_models.py         # ML model training
│   ├── shap_analysis.py        # SHAP explainability
│   ├── eda_visualizations.py   # EDA figures
│   ├── statistical_analysis.py # Statistical tests
│   └── visualize_model_results.py  # Model figures
│
├── 📂 streamlit_app/           # Interactive web app
│   └── app.py                  # 6-page dashboard
│
├── 📂 figures/                 # Publication figures (600 DPI)
│   ├── figure_01_sample_characteristics.png
│   ├── figure_07_roc_curves.png
│   └── ... (15 total)
│
├── 📂 results/                 # Analysis outputs
│   ├── model_comparison.csv
│   ├── shap_feature_importance.csv
│   ├── chi_square_tests.csv
│   └── table1_sample_characteristics.csv
│
├── 📂 models/                  # Trained models (excluded from repo)
├── 📂 data/                    # Data files (excluded from repo)
├── 📄 requirements.txt         # Python dependencies
└── 📄 README.md                # This file
```

---

## 🔬 Methodology

### 1. Data Preparation
- **Merging**: 3 NREL survey datasets integrated
- **Cleaning**: 95 high-missing variables dropped, sophisticated imputation
- **Engineering**: 12 derived features created

### 2. Machine Learning
- **Models**: Logistic Regression (baseline), Random Forest, XGBoost, LightGBM
- **Imbalance Handling**: Class weights + scale_pos_weight
- **Validation**: 70/30 stratified train-test split
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC

### 3. Explainable AI (XAI)
- **Method**: SHAP (SHapley Additive exPlanations)
- **Analyses**: 
  - Global feature importance
  - Individual prediction explanations
  - Feature dependence plots
  - Waterfall plots for case studies

### 4. Statistical Analysis
- **Tests**: Chi-square (categorical), t-tests (continuous), correlation
- **Effect Sizes**: Cohen's d, Cramér's V, Pearson's r
- **Reporting**: Table 1 (sample characteristics), all p-values < 0.001

---

## 📊 Sample Visualizations

<div align="center">

### Model Performance Comparison
*ROC curves showing XGBoost achieving near-perfect discrimination (AUC=0.976)*

### SHAP Feature Importance
*Top 20 features ranked by mean absolute SHAP value*

### Confusion Matrix
*XGBoost: 97.3% Non-EV accuracy, 86.5% EV recall*

</div>

---

## 🛠️ Technologies

### Core Libraries
- **ML/Stats**: `scikit-learn` `xgboost` `lightgbm` `scipy` `statsmodels`
- **Explainability**: `shap`
- **Visualization**: `matplotlib` `seaborn` `plotly`
- **Web App**: `streamlit`
- **Data**: `pandas` `numpy`

### Development
- **Python**: 3.9+
- **IDE**: VS Code (recommended)
- **Version Control**: Git/GitHub

---

## 📄 Publication

### Status
🚧 **Manuscript in Preparation**

**Target Journals**:
- Transport Policy (Elsevier)
- Energy Research & Social Science
- Transportation Research Part F

### Cite This Work
```bibtex
@misc{mahbub2025ev,
  title={Interpretable Machine Learning for Electric Vehicle Adoption: 
         A California Household Analysis},
  author={MAHBUB},
  year={2025},
  institution={Chulalongkorn University},
  howpublished={\url{https://github.com/mahbubchula/CA-EV-Adoption-Research}}
}
```

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

<div align="center">

**MAHBUB**

🏛️ Chulalongkorn University, Bangkok, Thailand  
📚 Department of Civil Engineering  
📧 6870376421@student.chula.ac.th  
🔗 [GitHub](https://github.com/mahbubchula)

</div>

---

## 🙏 Acknowledgments

- **Data Source**: NREL Transportation Secure Data Center
- **Institution**: Chulalongkorn University
- **Funding**: [If applicable]

Special thanks to the NREL team for providing access to the California Vehicle Survey data.

---

## 📜 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Data Usage
- **NREL Data**: Subject to TSDC data use agreement
- **Code**: Open source (MIT License)
- **Figures**: CC BY 4.0 (attribution required)

---

## 📌 Project Status

<div align="center">

### Development Timeline

| Phase | Status | Duration |
|:------|:------:|:--------:|
| ✅ Data Preparation | Complete | Week 1 |
| ✅ Exploratory Analysis | Complete | Week 2 |
| ✅ ML Modeling | Complete | Week 3 |
| ✅ SHAP Analysis | Complete | Week 4 |
| ✅ Streamlit Dashboard | Complete | Week 6 |
| 🚧 Manuscript Writing | In Progress | Week 7-10 |
| ⏳ Journal Submission | Planned | 2025 Q1 |

</div>

---

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐!

---

<div align="center">

**Made with ❤️ for sustainable transportation research**

*Last Updated: December 26, 2025*

</div>