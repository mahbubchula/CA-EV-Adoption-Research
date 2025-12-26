"""
Data cleaning and missing data handling
Author: MAHBUB
Date: December 26, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config import PROCESSED_DATA_PATHS, VARS_TO_DROP, RANDOM_STATE


def analyze_missing_data(df):
    """
    Comprehensive missing data analysis.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Missing data report
    """
    print("="*80)
    print("MISSING DATA ANALYSIS")
    print("="*80)
    
    missing_report = pd.DataFrame({
        'Variable': df.columns,
        'N_Missing': df.isnull().sum(),
        'Pct_Missing': (df.isnull().sum() / len(df) * 100).round(2),
        'N_Complete': df.notnull().sum(),
        'Data_Type': df.dtypes
    }).sort_values('Pct_Missing', ascending=False)
    
    # Categorize by missingness
    missing_report['Category'] = missing_report['Pct_Missing'].apply(
        lambda x: 'DROP (>90%)' if x > 90 
        else 'HIGH (50-90%)' if x > 50
        else 'MODERATE (10-50%)' if x > 10
        else 'LOW (<10%)' if x > 0
        else 'COMPLETE'
    )
    
    # Print summary
    print("\nMissing Data Summary by Category:")
    print(missing_report['Category'].value_counts().sort_index())
    
    print("\n" + "-"*80)
    print("Variables with >50% missing (top 20):")
    print("-"*80)
    top_missing = missing_report[missing_report['Pct_Missing'] > 50].head(20)
    print(top_missing[['Variable', 'Pct_Missing', 'Category']].to_string(index=False))
    
    # Save report
    report_path = PROCESSED_DATA_PATHS['missing_report']
    missing_report.to_csv(report_path, index=False)
    print(f"\n✓ Full missing data report saved to: {report_path}")
    
    return missing_report


def drop_high_missing_vars(df, threshold=90):
    """
    Drop variables with >threshold% missing data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        threshold (float): Percentage threshold (default 90%)
        
    Returns:
        pd.DataFrame: Cleaned dataframe
        list: Dropped variable names
    """
    print("\n" + "="*80)
    print(f"DROPPING VARIABLES WITH >{threshold}% MISSING")
    print("="*80)
    
    # Calculate missing percentages
    missing_pct = (df.isnull().sum() / len(df) * 100)
    
    # Find variables to drop
    vars_to_drop_auto = missing_pct[missing_pct > threshold].index.tolist()
    
    # Combine with manual drop list from config
    all_vars_to_drop = list(set(vars_to_drop_auto + VARS_TO_DROP))
    
    # Remove if not in dataframe
    all_vars_to_drop = [v for v in all_vars_to_drop if v in df.columns]
    
    print(f"\nDropping {len(all_vars_to_drop)} variables:")
    for var in sorted(all_vars_to_drop):
        missing = missing_pct.get(var, 0)
        print(f"  • {var:40s} ({missing:.1f}% missing)")
    
    # Drop
    df_clean = df.drop(columns=all_vars_to_drop)
    
    print(f"\n✓ Remaining features: {df_clean.shape[1]} (dropped {len(all_vars_to_drop)})")
    
    return df_clean, all_vars_to_drop


def impute_numerical(df, strategy='grouped_median'):
    """
    Impute missing numerical variables.
    
    Args:
        df (pd.DataFrame): Input dataframe
        strategy (str): 'grouped_median' or 'simple_median'
        
    Returns:
        pd.DataFrame: Dataframe with imputed values
    """
    print("\n" + "="*80)
    print("IMPUTING NUMERICAL VARIABLES")
    print("="*80)
    
    # Identify numerical variables with missing data
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    missing_numerical = [col for col in numerical_cols if df[col].isnull().sum() > 0]
    
    print(f"\nFound {len(missing_numerical)} numerical variables with missing data")
    
    # Key variables to impute with grouped strategy
    grouped_impute_vars = {
        'acquired_price_1': 'veh_class_nrel',
        'acquired_price_2': 'veh_class_nrel',
        'mpg_1': 'veh_class_nrel',
        'mpg_2': 'veh_class_nrel',
        'annual_mileage': 'veh_class_nrel'
    }
    
    for var in missing_numerical:
        missing_count = df[var].isnull().sum()
        missing_pct = (missing_count / len(df) * 100)
        
        if missing_pct > 90:  # Skip if too much missing (should be dropped already)
            continue
        
        print(f"\n  {var} ({missing_pct:.1f}% missing)")
        
        if var in grouped_impute_vars and strategy == 'grouped_median':
            # Grouped imputation
            group_col = grouped_impute_vars[var]
            if group_col in df.columns:
                print(f"    → Grouped median imputation by {group_col}")
                df[var] = df.groupby(group_col)[var].transform(
                    lambda x: x.fillna(x.median())
                )
                
                # If still missing (group had all NaN), use global median
                if df[var].isnull().sum() > 0:
                    global_median = df[var].median()
                    df[var] = df[var].fillna(global_median)
                    print(f"    → Remaining filled with global median: {global_median:.2f}")
            else:
                # Group column not available, use simple median
                median_val = df[var].median()
                df[var] = df[var].fillna(median_val)
                print(f"    → Simple median imputation: {median_val:.2f}")
        else:
            # Simple median imputation
            median_val = df[var].median()
            df[var] = df[var].fillna(median_val)
            print(f"    → Simple median imputation: {median_val:.2f}")
        
        print(f"    ✓ Remaining missing: {df[var].isnull().sum()}")
    
    return df


def impute_binary_experience(df):
    """
    Impute binary EV experience variables.
    Assumption: Missing = No experience (conservative)
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with imputed binary variables
    """
    print("\n" + "="*80)
    print("IMPUTING BINARY EV EXPERIENCE VARIABLES")
    print("="*80)
    
    # Experience variables
    binary_vars = [
        'hybrid_experience', 'phev_experience', 'bev_experience',
        'past_hybrid', 'past_phev', 'past_bev', 'fcv_experience'
    ]
    
    print("\nAssumption: Missing = No experience (0)")
    
    for var in binary_vars:
        if var in df.columns:
            missing_count = df[var].isnull().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(df) * 100)
                print(f"  • {var:25s}: {missing_count:6,} missing ({missing_pct:5.1f}%) → filling with 0")
                df[var] = df[var].fillna(0)
    
    return df


def impute_categorical(df, strategy='mode'):
    """
    Impute categorical variables.
    
    Args:
        df (pd.DataFrame): Input dataframe
        strategy (str): 'mode' or 'unknown'
        
    Returns:
        pd.DataFrame: Dataframe with imputed categorical values
    """
    print("\n" + "="*80)
    print("IMPUTING CATEGORICAL VARIABLES")
    print("="*80)
    
    # Identify categorical variables with missing data
    categorical_cols = df.select_dtypes(include=['object']).columns
    missing_categorical = [col for col in categorical_cols if df[col].isnull().sum() > 0]
    
    print(f"\nFound {len(missing_categorical)} categorical variables with missing data")
    
    for var in missing_categorical:
        missing_count = df[var].isnull().sum()
        missing_pct = (missing_count / len(df) * 100)
        
        if missing_pct > 90:  # Skip if already dropped
            continue
        
        print(f"\n  {var} ({missing_pct:.1f}% missing)")
        
        if missing_pct > 50:
            # High missingness: create "Unknown" category
            df[var] = df[var].fillna('Unknown')
            print(f"    → Added 'Unknown' category")
        else:
            # Low/moderate missingness: use mode
            mode_val = df[var].mode()[0] if len(df[var].mode()) > 0 else 'Unknown'
            df[var] = df[var].fillna(mode_val)
            print(f"    → Mode imputation: '{mode_val}'")
        
        print(f"    ✓ Remaining missing: {df[var].isnull().sum()}")
    
    return df


def drop_rows_with_missing_target(df, target_var='is_ev'):
    """
    Drop rows where target variable is missing.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_var (str): Target variable name
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    print("\n" + "="*80)
    print("REMOVING RECORDS WITH MISSING TARGET")
    print("="*80)
    
    initial_count = len(df)
    missing_target = df[target_var].isnull().sum()
    
    if missing_target > 0:
        print(f"\nFound {missing_target:,} records with missing target variable")
        df = df[df[target_var].notna()].copy()
        print(f"  ✓ Dropped {missing_target:,} records")
        print(f"  ✓ Remaining: {len(df):,} records")
    else:
        print("\n✓ No missing values in target variable")
    
    return df


def final_data_check(df):
    """
    Final check for any remaining missing data.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe
        
    Returns:
        pd.DataFrame: Summary of remaining missing data
    """
    print("\n" + "="*80)
    print("FINAL DATA QUALITY CHECK")
    print("="*80)
    
    remaining_missing = df.isnull().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]
    
    if len(remaining_missing) > 0:
        print(f"\n⚠ WARNING: {len(remaining_missing)} variables still have missing data:")
        for var, count in remaining_missing.items():
            pct = (count / len(df) * 100)
            print(f"  • {var:40s}: {count:6,} ({pct:5.1f}%)")
    else:
        print("\n✓ No missing data remaining!")
    
    # Overall statistics
    print(f"\nFinal Dataset Statistics:")
    print(f"  • Total records: {len(df):,}")
    print(f"  • Total features: {df.shape[1]}")
    print(f"  • Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return remaining_missing


def main():
    """
    Main function to run complete data cleaning pipeline.
    """
    # Load merged data
    print("Loading merged data...")
    merged_path = PROCESSED_DATA_PATHS['merged_data']
    
    if not merged_path.exists():
        print(f"\n✗ ERROR: Merged data not found at {merged_path}")
        print("  Please run data_loader.py first to create merged dataset.")
        return None
    
    df = pd.read_csv(merged_path, low_memory=False)
    print(f"  ✓ Loaded {len(df):,} records × {df.shape[1]} features")
    
    # Step 1: Analyze missing data
    missing_report = analyze_missing_data(df)
    
    # Step 2: Drop high missing variables
    df, dropped_vars = drop_high_missing_vars(df, threshold=90)
    
    # Step 3: Drop rows with missing target
    df = drop_rows_with_missing_target(df)
    
    # Step 4: Impute numerical variables
    df = impute_numerical(df, strategy='grouped_median')
    
    # Step 5: Impute binary experience variables
    df = impute_binary_experience(df)
    
    # Step 6: Impute categorical variables
    df = impute_categorical(df, strategy='mode')
    
    # Step 7: Final check
    remaining_missing = final_data_check(df)
    
    # Save cleaned data
    cleaned_path = PROCESSED_DATA_PATHS['cleaned_data']
    print(f"\nSaving cleaned data to: {cleaned_path}")
    df.to_csv(cleaned_path, index=False)
    print(f"  ✓ Saved successfully")
    
    print("\n" + "="*80)
    print("✓ DATA CLEANING COMPLETE")
    print("="*80)
    print(f"\nCleaned dataset saved to: {cleaned_path}")
    print(f"Next step: Run feature_engineering.py to create derived features")
    
    return df


if __name__ == "__main__":
    cleaned_data = main()