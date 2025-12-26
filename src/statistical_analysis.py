"""
Statistical Analysis - Hypothesis Testing
Author: MAHBUB
Date: December 26, 2025
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
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

def chi_square_tests(df):
    """
    Perform chi-square tests for categorical variables.
    Tests association between categorical variables and EV adoption.
    """
    print("\n" + "="*80)
    print("CHI-SQUARE TESTS (Categorical Variables)")
    print("="*80)
    
    categorical_vars = [
        ('income_category', 'Income Category'),
        ('college_degree_plus', 'College Degree'),
        ('mileage_category', 'Mileage Category'),
        ('multi_vehicle_household', 'Multi-Vehicle Household'),
        ('urban_region', 'Urban Region'),
    ]
    
    results = []
    
    for var, label in categorical_vars:
        if var not in df.columns:
            continue
            
        # Create contingency table
        contingency = pd.crosstab(df[var], df['is_ev'])
        
        # Perform chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        # Calculate Cramér's V (effect size)
        n = contingency.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
        
        results.append({
            'Variable': label,
            'Chi2': chi2,
            'p-value': p_value,
            'DOF': dof,
            'Cramers_V': cramers_v,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })
        
        print(f"\n{label}:")
        print(f"  χ² = {chi2:.4f}, df = {dof}, p = {p_value:.4e}")
        print(f"  Cramér's V = {cramers_v:.4f} (effect size)")
        if p_value < 0.001:
            print(f"  *** Highly significant (p < 0.001)")
        elif p_value < 0.01:
            print(f"  ** Very significant (p < 0.01)")
        elif p_value < 0.05:
            print(f"  * Significant (p < 0.05)")
        else:
            print(f"  Not significant (p ≥ 0.05)")
    
    return pd.DataFrame(results)

def t_tests(df):
    """
    Perform independent t-tests for continuous variables.
    Compares means between EV and Non-EV owners.
    """
    print("\n" + "="*80)
    print("INDEPENDENT T-TESTS (Continuous Variables)")
    print("="*80)
    
    continuous_vars = [
        ('income', 'Income Level'),
        ('annual_mileage', 'Annual Mileage'),
        ('vehicle_age_approx', 'Vehicle Age'),
        ('ev_experience_score', 'EV Experience Score'),
        ('charging_access_index', 'Charging Access Index'),
        ('adoption_readiness_score', 'Adoption Readiness Score'),
    ]
    
    results = []
    
    for var, label in continuous_vars:
        if var not in df.columns:
            continue
        
        # Separate groups
        non_ev = df[df['is_ev'] == 0][var].dropna()
        ev = df[df['is_ev'] == 1][var].dropna()
        
        # Perform t-test
        t_stat, p_value = ttest_ind(non_ev, ev)
        
        # Calculate Cohen's d (effect size)
        pooled_std = np.sqrt(((len(non_ev)-1)*non_ev.std()**2 + 
                             (len(ev)-1)*ev.std()**2) / 
                            (len(non_ev) + len(ev) - 2))
        cohens_d = (ev.mean() - non_ev.mean()) / pooled_std
        
        results.append({
            'Variable': label,
            'Non-EV_Mean': non_ev.mean(),
            'EV_Mean': ev.mean(),
            'Difference': ev.mean() - non_ev.mean(),
            't-statistic': t_stat,
            'p-value': p_value,
            'Cohens_d': cohens_d,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })
        
        print(f"\n{label}:")
        print(f"  Non-EV mean: {non_ev.mean():.3f} (SD = {non_ev.std():.3f})")
        print(f"  EV mean:     {ev.mean():.3f} (SD = {ev.std():.3f})")
        print(f"  Difference:  {ev.mean() - non_ev.mean():.3f}")
        print(f"  t = {t_stat:.4f}, p = {p_value:.4e}")
        print(f"  Cohen's d = {cohens_d:.4f} (effect size)")
        if p_value < 0.001:
            print(f"  *** Highly significant (p < 0.001)")
        elif p_value < 0.01:
            print(f"  ** Very significant (p < 0.01)")
        elif p_value < 0.05:
            print(f"  * Significant (p < 0.05)")
        else:
            print(f"  Not significant (p ≥ 0.05)")
    
    return pd.DataFrame(results)

def correlation_analysis(df):
    """
    Correlation analysis between key variables and EV adoption.
    """
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS")
    print("="*80)
    
    continuous_vars = [
        'income',
        'annual_mileage',
        'vehicle_age_approx',
        'ev_experience_score',
        'charging_access_index',
        'adoption_readiness_score',
        'is_ev'
    ]
    
    # Select available variables
    available_vars = [v for v in continuous_vars if v in df.columns]
    
    # Calculate correlation matrix
    corr_matrix = df[available_vars].corr()
    
    # Focus on correlations with is_ev
    ev_corr = corr_matrix['is_ev'].drop('is_ev').sort_values(ascending=False)
    
    print("\nCorrelation with EV Adoption (is_ev):")
    print("-" * 60)
    for var, corr in ev_corr.items():
        # Classify correlation strength
        if abs(corr) > 0.5:
            strength = "Strong"
        elif abs(corr) > 0.3:
            strength = "Moderate"
        elif abs(corr) > 0.1:
            strength = "Weak"
        else:
            strength = "Very Weak"
        
        print(f"  {var:30s}: r = {corr:+.4f} ({strength})")
    
    return corr_matrix

def effect_size_interpretation():
    """
    Print interpretation guide for effect sizes.
    """
    print("\n" + "="*80)
    print("EFFECT SIZE INTERPRETATION GUIDE")
    print("="*80)
    
    print("\nCramér's V (Chi-square):")
    print("  • < 0.1  : Negligible")
    print("  • 0.1-0.3: Small")
    print("  • 0.3-0.5: Medium")
    print("  • > 0.5  : Large")
    
    print("\nCohen's d (t-test):")
    print("  • < 0.2  : Small")
    print("  • 0.2-0.5: Small to Medium")
    print("  • 0.5-0.8: Medium to Large")
    print("  • > 0.8  : Large")
    
    print("\nPearson's r (Correlation):")
    print("  • < 0.1  : Very Weak")
    print("  • 0.1-0.3: Weak")
    print("  • 0.3-0.5: Moderate")
    print("  • > 0.5  : Strong")

def save_results(chi2_results, ttest_results, corr_matrix):
    """
    Save statistical results to CSV files.
    """
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Create results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save chi-square results
    chi2_path = RESULTS_DIR / 'chi_square_tests.csv'
    chi2_results.to_csv(chi2_path, index=False)
    print(f"  ✓ Chi-square results: {chi2_path}")
    
    # Save t-test results
    ttest_path = RESULTS_DIR / 't_test_results.csv'
    ttest_results.to_csv(ttest_path, index=False)
    print(f"  ✓ T-test results: {ttest_path}")
    
    # Save correlation matrix
    corr_path = RESULTS_DIR / 'correlation_matrix.csv'
    corr_matrix.to_csv(corr_path)
    print(f"  ✓ Correlation matrix: {corr_path}")

def main():
    """Main function to run all statistical analyses."""
    print("="*80)
    print("WEEK 2: STATISTICAL ANALYSIS")
    print("="*80)
    
    # Load data
    df = load_data()
    
    # Run analyses
    chi2_results = chi_square_tests(df)
    ttest_results = t_tests(df)
    corr_matrix = correlation_analysis(df)
    
    # Print interpretation guide
    effect_size_interpretation()
    
    # Save results
    save_results(chi2_results, ttest_results, corr_matrix)
    
    print("\n" + "="*80)
    print("✓ STATISTICAL ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Findings Summary:")
    
    # Summarize significant results
    sig_chi2 = chi2_results[chi2_results['p-value'] < 0.05]
    sig_ttest = ttest_results[ttest_results['p-value'] < 0.05]
    
    print(f"\n• {len(sig_chi2)} categorical variables significantly associated with EV adoption")
    print(f"• {len(sig_ttest)} continuous variables significantly different between groups")
    
    if len(sig_ttest) > 0:
        print(f"\nStrongest predictors (by effect size):")
        top_predictors = sig_ttest.nlargest(3, 'Cohens_d')
        for idx, row in top_predictors.iterrows():
            print(f"  • {row['Variable']}: Cohen's d = {row['Cohens_d']:.3f}, p < 0.001")
    
    print("\nNext step: Run descriptive_stats.py to create Table 1 for paper")

if __name__ == "__main__":
    main()