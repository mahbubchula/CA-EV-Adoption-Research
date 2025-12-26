import pandas as pd

print("Loading data...")
df = pd.read_csv('data/processed/cleaned_data.csv')

print(f"\nBefore fix: {df.isnull().sum().sum()} missing values")

# Check which columns have missing
missing_cols = df.columns[df.isnull().any()].tolist()
print(f"\nColumns with missing data:")
for col in missing_cols:
    print(f"  - {col}: {df[col].isnull().sum()} missing")

# Fix missing values in engineered features
if 'income_category' in df.columns:
    df['income_category'] = df['income_category'].fillna('Medium')

if 'mileage_category' in df.columns:
    df['mileage_category'] = df['mileage_category'].fillna('Medium')

if 'affordability_ratio' in df.columns:
    df['affordability_ratio'] = df['affordability_ratio'].fillna(0)

if 'vehicle_age_approx' in df.columns:
    df['vehicle_age_approx'] = df['vehicle_age_approx'].fillna(df['vehicle_age_approx'].median())

if 'adoption_readiness_score' in df.columns:
    df['adoption_readiness_score'] = df['adoption_readiness_score'].fillna(df['adoption_readiness_score'].median())

# Fill any remaining missing with appropriate defaults
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(df[col].median())

print(f"\nAfter fix: {df.isnull().sum().sum()} missing values")

# Save
df.to_csv('data/processed/cleaned_data.csv', index=False)
print("\n✓ Fixed and saved!")