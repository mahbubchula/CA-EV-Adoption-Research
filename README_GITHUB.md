# Create README.md
@"
# 🚗 California EV Adoption Research

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Interpretable Machine Learning for Electric Vehicle Adoption Analysis**

Machine learning study analyzing EV adoption patterns among **7,353 California households** using XGBoost, SHAP explainability, and interactive visualizations.

---

## 🎯 Project Highlights

- **📊 Dataset**: 7,353 vehicles from 3,800 California households (NREL 2024)
- **🎯 Model Accuracy**: **95.97%** (XGBoost)
- **🔍 ROC-AUC**: **0.976** (Near-perfect discrimination)
- **📈 Figures**: 15 publication-grade visualizations (600 DPI)
- **📱 Web App**: Interactive 6-page Streamlit dashboard

---

## 🏆 Key Results

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Logistic Regression | 88.98% | 0.593 | 0.884 |
| Random Forest | 81.10% | 0.530 | 0.909 |
| **XGBoost** ⭐ | **95.97%** | **0.838** | **0.976** |
| LightGBM | 95.92% | 0.838 | 0.975 |

### Top Predictors (SHAP)
1. 💰 **Income Level** (Cohen's d = 0.48)
2. 🎓 **Education** (94% EV owners have college degree)
3. ⏰ **Vehicle Age** (EVs 3 years newer)

---

## 📁 Project Structure

\`\`\`
CA_EV_Research/
├── src/                    # Source code
│   ├── data_loader.py
│   ├── train_models.py
│   └── shap_analysis.py
├── streamlit_app/          # Interactive dashboard
│   └── app.py
├── figures/                # 15 publication figures
├── results/                # Analysis outputs
└── requirements.txt
\`\`\`

---

## 🚀 Quick Start

### 1. Clone Repository
\`\`\`bash
git clone https://github.com/mahbubchula/CA-EV-Adoption-Research.git
cd CA-EV-Adoption-Research
\`\`\`

### 2. Install Dependencies
\`\`\`bash
python -m venv venv
venv\\Scripts\\activate  # Windows
pip install -r requirements.txt
\`\`\`

### 3. Download Data
⚠️ Data not included (too large). Download from [NREL TSDC](https://www.nrel.gov/transportation/secure-transportation-data/)

Place in: \`data/raw/data/\`

### 4. Run Pipeline
\`\`\`bash
cd src
python data_loader.py
python data_cleaning.py
python train_models.py
python shap_analysis.py
\`\`\`

### 5. Launch Dashboard
\`\`\`bash
streamlit run streamlit_app/app.py
\`\`\`

---

## 📱 Interactive Dashboard

6-page Streamlit application:
- 🔮 **EV Predictor** - Individual predictions
- 📊 **Model Performance** - Metrics & comparisons
- 🔍 **Feature Importance** - SHAP analysis
- 💡 **Policy Simulator** - What-if scenarios
- 📈 **Data Explorer** - Interactive filtering

---

## 🔬 Methodology

- **Data**: NREL California Vehicle Survey 2024
- **Sample**: 7,353 vehicles, 3,800 households
- **Models**: Logistic Regression, Random Forest, XGBoost, LightGBM
- **Explainability**: SHAP TreeExplainer
- **Validation**: 70/30 stratified split

---

## 📊 Sample Figures

*15 publication-quality figures (600 DPI) included in \`figures/\` directory*

---

## 🛠️ Tech Stack

- Python 3.9+
- scikit-learn, XGBoost, LightGBM
- SHAP (explainability)
- Streamlit (web app)
- matplotlib, seaborn, plotly

---

## 📧 Contact

**MAHBUB**  
🏛️ Chulalongkorn University  
📧 6870376421@student.chula.ac.th  
🔗 [GitHub](https://github.com/mahbubchula)

---

## 📜 License

MIT License - See [LICENSE](LICENSE) file

---

## 🙏 Acknowledgments

- **Data**: NREL Transportation Secure Data Center
- **Institution**: Chulalongkorn University

---

⭐ **Star this repo if you find it useful!**

*Last Updated: December 26, 2025*
"@ | Out-File -FilePath README.md -Encoding UTF8