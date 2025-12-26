# 📂 PROJECT FOLDER SETUP

## COPY THIS EXACT STRUCTURE TO YOUR LOCAL MACHINE

```
📁 CA_EV_Research/                          ← Your project root folder
│
├── 📄 run_week1.py                         ← Master script (DOWNLOAD THIS)
├── 📄 requirements.txt                     ← Python packages (DOWNLOAD THIS)
├── 📄 README.md                            ← Full documentation (DOWNLOAD THIS)
├── 📄 QUICKSTART.md                        ← Quick guide (DOWNLOAD THIS)
│
├── 📁 data/
│   ├── 📁 raw/                             ← PUT YOUR DATASET HERE
│   │   ├── 📁 data/
│   │   │   ├── residential_vehicle.csv     ← From your ZIP file
│   │   │   ├── residential_background.csv  ← From your ZIP file
│   │   │   ├── residential_household.csv   ← From your ZIP file
│   │   │   ├── commercial_vehicle.csv
│   │   │   └── commercial_background.csv
│   │   └── 📁 documentation/
│   │       └── California_vehicle_survey_data_dictionary_2024.xlsx
│   │
│   └── 📁 processed/                       ← Auto-generated outputs
│       ├── merged_residential_data.csv     (created by run_week1.py)
│       ├── cleaned_data.csv                (created by run_week1.py)
│       └── missing_data_report.csv         (created by run_week1.py)
│
├── 📁 src/                                 ← PUT ALL .PY CODE FILES HERE
│   ├── 📄 config.py                        (DOWNLOAD THIS)
│   ├── 📄 data_loader.py                   (DOWNLOAD THIS)
│   ├── 📄 data_cleaning.py                 (DOWNLOAD THIS)
│   └── 📄 feature_engineering.py           (DOWNLOAD THIS)
│
├── 📁 figures/                             ← Week 2+ outputs
├── 📁 results/                             ← Week 3+ outputs
├── 📁 models/                              ← Week 3+ outputs
├── 📁 notebooks/                           ← Your Jupyter notebooks (optional)
└── 📁 docs/                                ← Additional documentation
```

---

## ⚡ 3-STEP SETUP (DO THIS FIRST!)

### **STEP 1: Create Folder Structure**

**Option A - Command Line (Fastest):**
```bash
mkdir -p CA_EV_Research/{data/{raw/{data,documentation},processed},src,figures,results,models,notebooks,docs}
cd CA_EV_Research
```

**Option B - Manually:**
1. Create folder `CA_EV_Research` on your Desktop
2. Inside it, create folders: `data`, `src`, `figures`, `results`, `models`
3. Inside `data`, create: `raw` and `processed`
4. Inside `raw`, create: `data` and `documentation`

### **STEP 2: Copy Your Dataset**

1. **Extract your ZIP file**
2. **Copy CSVs** to: `CA_EV_Research/data/raw/data/`
3. **Copy Excel file** to: `CA_EV_Research/data/raw/documentation/`

**Verify:**
```bash
ls data/raw/data/
# Should show: residential_vehicle.csv, residential_background.csv, etc.
```

### **STEP 3: Download Code Files**

**Put in PROJECT ROOT** (CA_EV_Research/):
- ✅ `run_week1.py`
- ✅ `requirements.txt`
- ✅ `README.md`
- ✅ `QUICKSTART.md`

**Put in src/** folder (CA_EV_Research/src/):
- ✅ `config.py`
- ✅ `data_loader.py`
- ✅ `data_cleaning.py`
- ✅ `feature_engineering.py`

**Final check:**
```bash
# You should have:
CA_EV_Research/
├── run_week1.py          ← ✓
├── requirements.txt      ← ✓
├── src/
│   ├── config.py         ← ✓
│   ├── data_loader.py    ← ✓
│   └── ...               ← ✓
└── data/
    └── raw/
        └── data/
            ├── residential_vehicle.csv  ← ✓
            └── ...                      ← ✓
```

---

## 🚀 INSTALL & RUN

### **Install Python Packages:**
```bash
cd CA_EV_Research
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### **Run Week 1:**
```bash
python run_week1.py
```

---

## 📊 EXPECTED OUTPUT

### **Console Output:**
```
================================================================================
CALIFORNIA EV RESEARCH - WEEK 1 PIPELINE
Data Preparation: Merging → Cleaning → Feature Engineering
================================================================================

🔄 STEP 1/3: DATA LOADING & MERGING
--------------------------------------------------------------------------------
Loading residential_vehicle...
  ✓ Shape: 7,353 rows × 29 columns
  ✓ Unique households: 3,800
...
✅ Step 1 Complete!

🧹 STEP 2/3: DATA CLEANING
--------------------------------------------------------------------------------
...
✅ Step 2 Complete!

⚙️  STEP 3/3: FEATURE ENGINEERING
--------------------------------------------------------------------------------
...
✅ Step 3 Complete!

================================================================================
✅ WEEK 1 PIPELINE COMPLETE!
================================================================================

📊 Final Dataset Summary:
   • Records: 6,951
   • Features: 322
   • EV Adoption Rate: 12.1%
   • Missing Data: 0 values

📁 Output Files:
   • Merged Data: data/processed/merged_residential_data.csv
   • Cleaned Data: data/processed/cleaned_data.csv
   • Missing Report: data/processed/missing_data_report.csv

⏱️  Total Time: 2.3 minutes
🎉 READY FOR WEEK 2: EXPLORATORY DATA ANALYSIS!
```

### **Files Created:**
```
data/processed/
├── merged_residential_data.csv  (~30 MB)
├── cleaned_data.csv             (~27 MB)
└── missing_data_report.csv      (~50 KB)
```

---

## ✅ VERIFICATION CHECKLIST

**After running, verify everything worked:**

```bash
# Check files exist
ls data/processed/
# Should show: merged_residential_data.csv, cleaned_data.csv, missing_data_report.csv

# Check file sizes
ls -lh data/processed/
# cleaned_data.csv should be ~20-30 MB

# Quick data check
python << EOF
import pandas as pd
df = pd.read_csv('data/processed/cleaned_data.csv')
print(f"✓ {len(df):,} records")
print(f"✓ {df.shape[1]} features")
print(f"✓ {df['is_ev'].sum():,} EVs ({df['is_ev'].mean()*100:.1f}%)")
print("\nNew features:")
print(df.columns[-10:].tolist())
EOF
```

**You should see:**
```
✓ 6,951 records
✓ 322 features
✓ 888 EVs (12.8%)

New features:
['ev_experience_score', 'charging_access_index', 'income_category', 
 'college_degree_plus', 'vehicle_age_approx', 'affordability_ratio',
 'mileage_category', 'multi_vehicle_household', 'urban_region', 
 'adoption_readiness_score']
```

---

## 🎯 WEEK 1 SUCCESS CRITERIA

| Criterion | Target | Status |
|-----------|--------|--------|
| **Dataset loaded** | 7,353 vehicles | ⬜ |
| **Data cleaned** | <5% missing | ⬜ |
| **Target variable** | `is_ev` exists | ⬜ |
| **Adoption rate** | ~12.1% | ⬜ |
| **New features** | 10+ created | ⬜ |
| **Output files** | 3 CSVs created | ⬜ |
| **Runtime** | <5 minutes | ⬜ |

**Check ALL boxes before moving to Week 2!**

---

## 🆘 COMMON ERRORS & FIXES

### **Error 1: "No module named 'pandas'"**
```bash
# Fix: Install requirements
pip install -r requirements.txt
```

### **Error 2: "FileNotFoundError: data/raw/data/residential_vehicle.csv"**
```bash
# Fix: Check your data location
ls data/raw/data/
# Should show CSV files

# If empty:
# 1. Extract your ZIP file
# 2. Copy CSVs to: CA_EV_Research/data/raw/data/
```

### **Error 3: "ModuleNotFoundError: No module named 'config'"**
```bash
# Fix: Add src to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"  # Mac/Linux
$env:PYTHONPATH += ";$(Get-Location)\src"     # Windows

# Or run from project root:
cd CA_EV_Research
python run_week1.py
```

### **Error 4: "MemoryError" or "Killed"**
```bash
# Fix: Process smaller sample
# Edit data_loader.py, line 45:
df = pd.read_csv(path, low_memory=False, nrows=5000)

# Or close other programs and try again
```

---

## 📚 NEXT STEPS

### **After Week 1 Success:**

1. **✅ Explore Your Data**
   ```bash
   jupyter notebook
   # Open: notebooks/01_data_exploration.ipynb
   ```

2. **✅ Read the Cleaned Data**
   ```python
   import pandas as pd
   df = pd.read_csv('data/processed/cleaned_data.csv')
   df.head()
   df.describe()
   ```

3. **✅ Check Missing Data Report**
   ```bash
   # Open in Excel or:
   python -c "import pandas as pd; print(pd.read_csv('data/processed/missing_data_report.csv').head(20))"
   ```

4. **✅ Prepare for Week 2**
   - Install Jupyter: `pip install jupyter`
   - Create notebook: `jupyter notebook`
   - Start exploring!

---

## 🎓 LEARNING RESOURCES

### **For Beginners:**
- Python basics: https://www.learnpython.org/
- Pandas tutorial: https://pandas.pydata.org/docs/getting_started/intro_tutorials/
- VS Code setup: https://code.visualstudio.com/docs/python/python-tutorial

### **For Data Science:**
- Exploratory Data Analysis: https://www.kaggle.com/learn/data-visualization
- Feature Engineering: https://www.kaggle.com/learn/feature-engineering
- Machine Learning: https://www.kaggle.com/learn/intro-to-machine-learning

---

**📧 Questions? Check README.md or contact MAHBUB**

**🎉 CONGRATS ON COMPLETING WEEK 1! Ready for Week 2?**