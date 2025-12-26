"""
Descriptive Statistics Table (Table 1 for Paper)
Author: MAHBUB
Date: December 26, 2025
"""

import pandas as pd
import numpy as np
from scipy import stats
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from config import PROCESSED_DATA_PATHS, RESULTS_DIR

def load_data():
    """Load cleaned dataset."""
    print("Loading data...")
    df = pd.read_csv(PROCESSED_DATA_PATHS['cleaned_data'])
    print(f"  ✓ Loaded {len(df):,} records")
    return df

def create_table1(df):
    """
    Create Table 1: Sample Characteristics
    Standard format for academic papers.
    """
    print("\n" + "="*80)
    print("CREATING TABLE 1: SAMPLE CHARACTERISTICS")
    print("="*80)
    
    # Separate groups
    all_data = df
    non_ev = df[df['is_ev'] == 0]
    ev = df[df['is_ev'] == 1]
    
    table_data = []
    
    # SECTION 1: SAMPLE SIZE
    print("\n1. Sample Size")
    table_data.append({
        'Variable': 'Sample Size',
        'Category': 'N',
        'All': f"{len(all_data):,}",
        'Non-EV': f"{len(non_ev):,} ({len(non_ev)/len(all_data)*100:.1f}%)",
        'EV': f"{len(ev):,} ({len(ev)/len(all_data)*100:.1f}%)",
        'p-value': '-'
    })
    
    # SECTION 2: DEMOGRAPHICS (Categorical)
    print("2. Demographics (Categorical)")
    
    # Income Categories
    for cat in ['Low', 'Medium', 'High']:
        all_pct = (all_data['income_category'] == cat).sum()
        non_ev_pct = (non_ev['income_category'] == cat).sum()
        ev_pct = (ev['income_category'] == cat).sum()
        
        # Chi-square p-value
        from scipy.stats import chi2_contingency
        contingency = pd.crosstab(df['income_category'], df['is_ev'])
        chi2, p_val, _, _ = chi2_contingency(contingency)
        p_str = f"{p_val:.3f}" if p_val >= 0.001 else "< 0.001"
        
        table_data.append({
            'Variable': 'Income' if cat == 'Low' else '',
            'Category': f"  {cat}",
            'All': f"{all_pct:,} ({all_pct/len(all_data)*100:.1f}%)",
            'Non-EV': f"{non_ev_pct:,} ({non_ev_pct/len(non_ev)*100:.1f}%)",
            'EV': f"{ev_pct:,} ({ev_pct/len(ev)*100:.1f}%)",
            'p-value': p_str if cat == 'Low' else ''
        })
    
    # Education
    contingency = pd.crosstab(df['college_degree_plus'], df['is_ev'])
    chi2, p_val, _, _ = chi2_contingency(contingency)
    p_str = f"{p_val:.3f}" if p_val >= 0.001 else "< 0.001"
    
    for cat, label in [(0, 'No College'), (1, 'College+')]:
        all_pct = (all_data['college_degree_plus'] == cat).sum()
        non_ev_pct = (non_ev['college_degree_plus'] == cat).sum()
        ev_pct = (ev['college_degree_plus'] == cat).sum()
        
        table_data.append({
            'Variable': 'Education' if cat == 0 else '',
            'Category': f"  {label}",
            'All': f"{all_pct:,} ({all_pct/len(all_data)*100:.1f}%)",
            'Non-EV': f"{non_ev_pct:,} ({non_ev_pct/len(non_ev)*100:.1f}%)",
            'EV': f"{ev_pct:,} ({ev_pct/len(ev)*100:.1f}%)",
            'p-value': p_str if cat == 0 else ''
        })
    
    # Urban/Rural
    contingency = pd.crosstab(df['urban_region'], df['is_ev'])
    chi2, p_val, _, _ = chi2_contingency(contingency)
    p_str = f"{p_val:.3f}" if p_val >= 0.001 else "< 0.001"
    
    for cat, label in [(0, 'Rural'), (1, 'Urban')]:
        all_pct = (all_data['urban_region'] == cat).sum()
        non_ev_pct = (non_ev['urban_region'] == cat).sum()
        ev_pct = (ev['urban_region'] == cat).sum()
        
        table_data.append({
            'Variable': 'Location' if cat == 0 else '',
            'Category': f"  {label}",
            'All': f"{all_pct:,} ({all_pct/len(all_data)*100:.1f}%)",
            'Non-EV': f"{non_ev_pct:,} ({non_ev_pct/len(non_ev)*100:.1f}%)",
            'EV': f"{ev_pct:,} ({ev_pct/len(ev)*100:.1f}%)",
            'p-value': p_str if cat == 0 else ''
        })
    
    # SECTION 3: CONTINUOUS VARIABLES
    print("3. Continuous Variables")
    
    continuous_vars = [
        ('income', 'Income Level (1-11)'),
        ('annual_mileage', 'Annual Mileage (miles)'),
        ('vehicle_age_approx', 'Vehicle Age (years)'),
        ('ev_experience_score', 'EV Experience Score (0-2)'),
        ('charging_access_index', 'Charging Access Index (0-1)'),
        ('adoption_readiness_score', 'Adoption Readiness (0-10)')
    ]
    
    for var, label in continuous_vars:
        if var not in df.columns:
            continue
        
        # Calculate statistics
        all_mean = all_data[var].mean()
        all_std = all_data[var].std()
        non_ev_mean = non_ev[var].mean()
        non_ev_std = non_ev[var].std()
        ev_mean = ev[var].mean()
        ev_std = ev[var].std()
        
        # T-test
        from scipy.stats import ttest_ind
        t_stat, p_val = ttest_ind(non_ev[var].dropna(), ev[var].dropna())
        p_str = f"{p_val:.3f}" if p_val >= 0.001 else "< 0.001"
        
        table_data.append({
            'Variable': label,
            'Category': 'Mean ± SD',
            'All': f"{all_mean:.2f} ± {all_std:.2f}",
            'Non-EV': f"{non_ev_mean:.2f} ± {non_ev_std:.2f}",
            'EV': f"{ev_mean:.2f} ± {ev_std:.2f}",
            'p-value': p_str
        })
    
    return pd.DataFrame(table_data)

def format_table_for_paper(table_df):
    """
    Format table for publication (LaTeX/Word compatible).
    """
    print("\n" + "="*80)
    print("TABLE 1: SAMPLE CHARACTERISTICS")
    print("="*80)
    print()
    
    # Print formatted table
    print(f"{'Variable':<30} {'Category':<20} {'All (N=7,353)':<20} "
          f"{'Non-EV (N=6,465)':<20} {'EV (N=888)':<20} {'p-value':<10}")
    print("-" * 130)
    
    for _, row in table_df.iterrows():
        print(f"{row['Variable']:<30} {row['Category']:<20} {row['All']:<20} "
              f"{row['Non-EV']:<20} {row['EV']:<20} {row['p-value']:<10}")

def save_table(table_df):
    """Save table to CSV and text formats."""
    print("\n" + "="*80)
    print("SAVING TABLE 1")
    print("="*80)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save as CSV
    csv_path = RESULTS_DIR / 'table1_sample_characteristics.csv'
    table_df.to_csv(csv_path, index=False)
    print(f"  ✓ CSV format: {csv_path}")
    
    # Save as formatted text (with UTF-8 encoding)
    txt_path = RESULTS_DIR / 'table1_sample_characteristics.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:  # ← Added encoding='utf-8'
        f.write("Table 1: Sample Characteristics of California Vehicle Survey (2024)\n")
        f.write("="*130 + "\n\n")
        f.write(table_df.to_string(index=False))
    print(f"  ✓ Text format: {txt_path}")

def main():
    """Main function."""
    print("="*80)
    print("WEEK 2: DESCRIPTIVE STATISTICS TABLE")
    print("="*80)
    
    # Load data
    df = load_data()
    
    # Create Table 1
    table_df = create_table1(df)
    
    # Display formatted table
    format_table_for_paper(table_df)
    
    # Save table
    save_table(table_df)
    
    print("\n" + "="*80)
    print("✓ TABLE 1 COMPLETE")
    print("="*80)
    print("\nThis table is ready for your paper!")
    print("You can copy it directly into Word/LaTeX")
    print("\nFiles saved in: results/")

if __name__ == "__main__":
    main()