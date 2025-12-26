import pandas as pd

df = pd.read_csv('data/processed/cleaned_data.csv')

print('='*60)
print('WEEK 1 FINAL DATASET')
print('='*60)
print(f'✓ Records: {len(df):,}')
print(f'✓ Features: {df.shape[1]}')
print(f'✓ Missing: {df.isnull().sum().sum()}')
print(f'✓ EV count: {df["is_ev"].sum():,} ({df["is_ev"].mean()*100:.1f}%)')

print(f'\n✓ New features exist:')
new_features = [
    'ev_experience_score',
    'charging_access_index', 
    'income_category',
    'adoption_readiness_score'
]

for f in new_features:
    exists = "✓" if f in df.columns else "✗"
    print(f'  {exists} {f}')

print('='*60)
print('✅ WEEK 1 COMPLETE!')
print('='*60)