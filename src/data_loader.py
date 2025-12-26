"""
Data loading and merging functions
Author: MAHBUB
Date: December 26, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from config import RAW_DATA_PATHS, PROCESSED_DATA_PATHS, RANDOM_STATE


def load_raw_data():
    """
    Load all raw CSV files from NREL dataset.
    
    Returns:
        dict: Dictionary containing all dataframes
    """
    print("="*80)
    print("LOADING RAW DATA")
    print("="*80)
    
    data = {}
    
    for name, path in RAW_DATA_PATHS.items():
        if name == 'data_dictionary':  # Skip Excel file for now
            continue
            
        print(f"\nLoading {name}...")
        try:
            df = pd.read_csv(path, low_memory=False)
            data[name] = df
            print(f"  ✓ Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            
            # Check for sampno (merge key)
            if 'sampno' in df.columns:
                print(f"  ✓ Unique households: {df['sampno'].nunique():,}")
        except FileNotFoundError:
            print(f"  ✗ ERROR: File not found at {path}")
            print(f"     Please check that your data is in the correct location.")
            return None
    
    return data


def merge_residential_data(data):
    """
    Merge residential vehicle, background, and household datasets.
    
    Args:
        data (dict): Dictionary with raw dataframes
        
    Returns:
        pd.DataFrame: Merged dataset
    """
    print("\n" + "="*80)
    print("MERGING RESIDENTIAL DATASETS")
    print("="*80)
    
    # Extract residential dataframes
    res_veh = data['residential_vehicle'].copy()
    res_bg = data['residential_background'].copy()
    res_hh = data['residential_household'].copy()
    
    print(f"\nStarting with:")
    print(f"  • Vehicles:   {res_veh.shape[0]:,} records")
    print(f"  • Background: {res_bg.shape[0]:,} households")
    print(f"  • Household:  {res_hh.shape[0]:,} person records")
    
    # Step 1: Aggregate household members to household level
    print("\n1. Aggregating household member data...")
    
    hh_aggregated = res_hh.groupby('sampno').agg({
        'member_count': 'first',  # Same for all members in household
        'age': 'max',  # Maximum age in household (head of household proxy)
        'education': 'max',  # Highest education in household
        'gender': lambda x: x.mode()[0] if len(x) > 0 else np.nan,  # Most common
        'employment': lambda x: x.mode()[0] if len(x) > 0 else np.nan,
        'license': 'sum',  # Number of licensed drivers
    }).reset_index()
    
    # Rename for clarity
    hh_aggregated.columns = [
        'sampno', 'household_size', 'max_age', 'max_education',
        'primary_gender', 'primary_employment', 'num_licensed_drivers'
    ]
    
    print(f"  ✓ Aggregated to {hh_aggregated.shape[0]:,} households")
    
    # Step 2: Merge vehicles + background
    print("\n2. Merging vehicles + background...")
    merged = res_veh.merge(
        res_bg, 
        on='sampno', 
        how='left', 
        suffixes=('_veh', '_bg')
    )
    print(f"  ✓ Result: {merged.shape[0]:,} records × {merged.shape[1]} columns")
    
    # Check merge success
    merge_rate = (merged['sampno'].notna().sum() / len(merged)) * 100
    print(f"  ✓ Merge success rate: {merge_rate:.1f}%")
    
    # Step 3: Merge with household aggregated data
    print("\n3. Merging with household data...")
    merged = merged.merge(
        hh_aggregated,
        on='sampno',
        how='left'
    )
    print(f"  ✓ Final result: {merged.shape[0]:,} records × {merged.shape[1]} columns")
    
    # Final statistics
    print(f"\n" + "="*80)
    print("MERGE SUMMARY")
    print("="*80)
    print(f"  • Total vehicles: {len(merged):,}")
    print(f"  • Unique households: {merged['sampno'].nunique():,}")
    print(f"  • Average vehicles per household: {len(merged) / merged['sampno'].nunique():.2f}")
    print(f"  • Total features: {merged.shape[1]}")
    
    return merged


def create_target_variable(df):
    """
    Create binary target variable: is_ev
    1 = Electric vehicle (BEV or PHEV)
    0 = Gasoline vehicle
    
    Args:
        df (pd.DataFrame): Merged dataframe
        
    Returns:
        pd.DataFrame: Dataframe with target variable added
    """
    print("\n" + "="*80)
    print("CREATING TARGET VARIABLE")
    print("="*80)
    
    # Create binary target
    df['is_ev'] = df['fuel_type_nrel'].isin(['Electricity', 'Electricity & Gasoline']).astype(int)
    
    # Also create multi-class target
    df['fuel_category'] = df['fuel_type_nrel'].map({
        'Gasoline': 0,
        'Electricity & Gasoline': 1,  # PHEV
        'Electricity': 2  # BEV
    })
    
    # Statistics
    print(f"\nTarget variable distribution:")
    print(f"  • Non-EV (Gasoline): {(df['is_ev'] == 0).sum():,} ({(df['is_ev'] == 0).sum()/len(df)*100:.1f}%)")
    print(f"  • EV (BEV + PHEV):   {(df['is_ev'] == 1).sum():,} ({(df['is_ev'] == 1).sum()/len(df)*100:.1f}%)")
    
    print(f"\nDetailed breakdown:")
    print(df['fuel_type_nrel'].value_counts())
    
    # Check for missing
    missing_target = df['fuel_type_nrel'].isna().sum()
    if missing_target > 0:
        print(f"\n  ⚠ WARNING: {missing_target} records with missing fuel type")
        print(f"    These will be dropped in data cleaning step.")
    
    return df


def save_merged_data(df, filepath=None):
    """
    Save merged dataset to CSV.
    
    Args:
        df (pd.DataFrame): Merged dataframe
        filepath (Path): Optional custom filepath
    """
    if filepath is None:
        filepath = PROCESSED_DATA_PATHS['merged_data']
    
    print(f"\nSaving merged data to: {filepath}")
    df.to_csv(filepath, index=False)
    print(f"  ✓ Saved successfully ({len(df):,} records)")


def main():
    """
    Main function to load and merge all data.
    Run this script directly to create merged dataset.
    """
    # Load raw data
    data = load_raw_data()
    
    if data is None:
        print("\n✗ ERROR: Could not load data. Please check file paths.")
        return None
    
    # Merge residential data
    merged_df = merge_residential_data(data)
    
    # Create target variable
    merged_df = create_target_variable(merged_df)
    
    # Save
    save_merged_data(merged_df)
    
    print("\n" + "="*80)
    print("✓ DATA MERGING COMPLETE")
    print("="*80)
    print(f"\nMerged dataset saved to: {PROCESSED_DATA_PATHS['merged_data']}")
    print(f"Next step: Run data_cleaning.py to handle missing data")
    
    return merged_df


if __name__ == "__main__":
    merged_data = main()