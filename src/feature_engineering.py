"""
Feature engineering - Create derived features
Author: MAHBUB
Date: December 26, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config import PROCESSED_DATA_PATHS, INCOME_LABELS, EDUCATION_LABELS


def create_ev_experience_score(df):
    """
    Create composite EV experience score (0-3 scale).
    Higher score = more EV experience
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new feature
    """
    print("\n1. Creating EV Experience Score...")
    
    # Weighted average of different EV types
    df['ev_experience_score'] = (
        df['hybrid_experience'].fillna(0) * 0.5 +
        df['phev_experience'].fillna(0) * 1.0 +
        df['bev_experience'].fillna(0) * 1.5
    ) / 3.0
    
    print(f"   ✓ Range: {df['ev_experience_score'].min():.2f} to {df['ev_experience_score'].max():.2f}")
    print(f"   ✓ Mean: {df['ev_experience_score'].mean():.2f}")
    
    return df


def create_charging_access_index(df):
    """
    Create composite charging access index (0-1 scale).
    Combines home charging, work charging, and public station awareness.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new feature
    """
    print("\n2. Creating Charging Access Index...")
    
    # Convert to binary if needed
    home_charge = df['home_charge'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0}).fillna(0)
    charge_work = df['charge_work'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0}).fillna(0)
    charge_spots = (df['charge_spots'] > 0).astype(int) if 'charge_spots' in df.columns else 0
    
    # Weighted combination
    df['charging_access_index'] = (
        home_charge * 0.5 +
        charge_work * 0.3 +
        charge_spots * 0.2
    )
    
    print(f"   ✓ Range: {df['charging_access_index'].min():.2f} to {df['charging_access_index'].max():.2f}")
    print(f"   ✓ Mean: {df['charging_access_index'].mean():.2f}")
    
    return df


def create_income_categories(df):
    """
    Create simplified income categories (Low/Medium/High).
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new feature
    """
    print("\n3. Creating Income Categories...")
    
    df['income_category'] = pd.cut(
        df['income'],
        bins=[0, 4, 7, 12],
        labels=['Low', 'Medium', 'High'],
        include_lowest=True
    )
    
    print(f"   ✓ Distribution:")
    print(df['income_category'].value_counts().to_string())
    
    return df


def create_education_binary(df):
    """
    Create binary education variable (College degree or higher).
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new feature
    """
    print("\n4. Creating Education Binary...")
    
    # Bachelor's degree (5) or higher = 1
    if 'max_education' in df.columns:
        df['college_degree_plus'] = (df['max_education'] >= 5).astype(int)
        print(f"   ✓ College degree or higher: {df['college_degree_plus'].sum():,} ({df['college_degree_plus'].mean()*100:.1f}%)")
    
    return df


def create_vehicle_age(df):
    """
    Create vehicle age approximation.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new feature
    """
    print("\n5. Creating Vehicle Age...")
    
    current_year = 2024
    
    if 'acquired_year' in df.columns:
        df['vehicle_age_approx'] = current_year - df['acquired_year']
        
        # Cap at reasonable values
        df['vehicle_age_approx'] = df['vehicle_age_approx'].clip(0, 30)
        
        print(f"   ✓ Range: {df['vehicle_age_approx'].min():.0f} to {df['vehicle_age_approx'].max():.0f} years")
        print(f"   ✓ Mean: {df['vehicle_age_approx'].mean():.1f} years")
    else:
        print("   ⚠ 'acquired_year' not found, skipping")
    
    return df


def create_affordability_ratio(df):
    """
    Create affordability ratio (vehicle price / income).
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new feature
    """
    print("\n6. Creating Affordability Ratio...")
    
    if 'acquired_price_1' in df.columns and 'income' in df.columns:
        # Convert income category to approximate dollars (midpoint of range)
        income_dollars = df['income'].map({
            1: 5000, 2: 15000, 3: 27500, 4: 42500,
            5: 62500, 6: 87500, 7: 112500, 8: 137500,
            9: 175000, 10: 225000, 11: 275000
        })
        
        df['affordability_ratio'] = df['acquired_price_1'] / income_dollars
        
        # Cap at reasonable values
        df['affordability_ratio'] = df['affordability_ratio'].clip(0, 3)
        
        print(f"   ✓ Range: {df['affordability_ratio'].min():.2f} to {df['affordability_ratio'].max():.2f}")
        print(f"   ✓ Mean: {df['affordability_ratio'].mean():.2f}")
    else:
        print("   ⚠ Required columns not found, skipping")
    
    return df


def create_mileage_categories(df):
    """
    Create mileage categories (Low/Medium/High).
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new feature
    """
    print("\n7. Creating Mileage Categories...")
    
    if 'annual_mileage' in df.columns:
        df['mileage_category'] = pd.cut(
            df['annual_mileage'],
            bins=[0, 7500, 15000, 100000],
            labels=['Low', 'Medium', 'High']
        )
        
        print(f"   ✓ Distribution:")
        print(df['mileage_category'].value_counts().to_string())
        
        # Also create binary for high mileage
        df['high_mileage_driver'] = (df['annual_mileage'] > 15000).astype(int)
    
    return df


def create_household_features(df):
    """
    Create household-related features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new features
    """
    print("\n8. Creating Household Features...")
    
    # Multi-vehicle household
    if 'vehicle_count' in df.columns:
        df['multi_vehicle_household'] = (df['vehicle_count'] > 1).astype(int)
        print(f"   ✓ Multi-vehicle households: {df['multi_vehicle_household'].sum():,} ({df['multi_vehicle_household'].mean()*100:.1f}%)")
    
    # Large household (>4 members)
    if 'household_size' in df.columns:
        df['large_household'] = (df['household_size'] > 4).astype(int)
        print(f"   ✓ Large households: {df['large_household'].sum():,} ({df['large_household'].mean()*100:.1f}%)")
    
    return df


def create_regional_features(df):
    """
    Create region-based features.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new features
    """
    print("\n9. Creating Regional Features...")
    
    if 'region' in df.columns:
        # Urban regions (Bay Area, LA, SD)
        urban_regions = ['Bay Area', 'Los Angeles', 'San Diego', 'Sacramento']
        df['urban_region'] = df['region'].isin(urban_regions).astype(int)
        
        print(f"   ✓ Urban region residents: {df['urban_region'].sum():,} ({df['urban_region'].mean()*100:.1f}%)")
    
    return df


def create_adoption_readiness_score(df):
    """
    Create composite EV adoption readiness score.
    Combines multiple factors into single 0-10 metric.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with new feature
    """
    print("\n10. Creating Adoption Readiness Score...")
    
    # Normalize each component to 0-1 scale
    components = []
    weights = []
    
    # EV Experience (weight: 0.3)
    if 'ev_experience_score' in df.columns:
        components.append(df['ev_experience_score'] / 3.0)
        weights.append(0.3)
    
    # Charging Access (weight: 0.3)
    if 'charging_access_index' in df.columns:
        components.append(df['charging_access_index'])
        weights.append(0.3)
    
    # Income (weight: 0.2)
    if 'income' in df.columns:
        components.append(df['income'] / 11.0)
        weights.append(0.2)
    
    # Education (weight: 0.2)
    if 'max_education' in df.columns:
        components.append(df['max_education'] / 8.0)
        weights.append(0.2)
    
    # Calculate weighted average
    if len(components) > 0:
        weights = np.array(weights) / sum(weights)  # Normalize weights
        df['adoption_readiness_score'] = sum(w * c for w, c in zip(weights, components)) * 10
        
        print(f"   ✓ Range: {df['adoption_readiness_score'].min():.2f} to {df['adoption_readiness_score'].max():.2f}")
        print(f"   ✓ Mean: {df['adoption_readiness_score'].mean():.2f}")
    
    return df


def main():
    """
    Main function to run complete feature engineering pipeline.
    """
    # Load cleaned data
    print("="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    cleaned_path = PROCESSED_DATA_PATHS['cleaned_data']
    
    if not cleaned_path.exists():
        print(f"\n✗ ERROR: Cleaned data not found at {cleaned_path}")
        print("  Please run data_cleaning.py first.")
        return None
    
    print("\nLoading cleaned data...")
    df = pd.read_csv(cleaned_path, low_memory=False)
    print(f"  ✓ Loaded {len(df):,} records × {df.shape[1]} features")
    
    initial_features = df.shape[1]
    
    # Create all engineered features
    df = create_ev_experience_score(df)
    df = create_charging_access_index(df)
    df = create_income_categories(df)
    df = create_education_binary(df)
    df = create_vehicle_age(df)
    df = create_affordability_ratio(df)
    df = create_mileage_categories(df)
    df = create_household_features(df)
    df = create_regional_features(df)
    df = create_adoption_readiness_score(df)
    
    final_features = df.shape[1]
    new_features = final_features - initial_features
    
    # Save
    output_path = PROCESSED_DATA_PATHS['cleaned_data']  # Overwrite cleaned data
    print(f"\nSaving enhanced dataset to: {output_path}")
    df.to_csv(output_path, index=False)
    print(f"  ✓ Saved successfully")
    
    print("\n" + "="*80)
    print("✓ FEATURE ENGINEERING COMPLETE")
    print("="*80)
    print(f"\nSummary:")
    print(f"  • Initial features: {initial_features}")
    print(f"  • New features created: {new_features}")
    print(f"  • Total features: {final_features}")
    print(f"\nNext step: Run exploratory_analysis.py to visualize the data")
    
    return df


if __name__ == "__main__":
    enhanced_data = main()