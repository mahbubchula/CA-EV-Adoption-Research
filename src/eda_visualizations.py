"""
Exploratory Data Analysis - Publication-Grade Visualizations
Author: MAHBUB
Date: December 26, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from config import PROCESSED_DATA_PATHS, FIGURES_DIR, FIGURE_SETTINGS, COLOR_PALETTE

# Set publication-quality defaults
plt.rcParams['font.family'] = FIGURE_SETTINGS['font_family']
plt.rcParams['font.size'] = FIGURE_SETTINGS['font_size']
plt.rcParams['figure.dpi'] = FIGURE_SETTINGS['dpi']
sns.set_style("whitegrid")
sns.set_palette("colorblind")

def load_data():
    """Load cleaned dataset."""
    print("Loading cleaned data...")
    df = pd.read_csv(PROCESSED_DATA_PATHS['cleaned_data'])
    print(f"  ✓ Loaded {len(df):,} records × {df.shape[1]} features")
    return df

def save_figure(fig, filename, tight=True):
    """Save figure with publication settings."""
    filepath = FIGURES_DIR / filename
    if tight:
        fig.savefig(filepath, dpi=FIGURE_SETTINGS['dpi'], 
                   bbox_inches=FIGURE_SETTINGS['bbox_inches'],
                   format=FIGURE_SETTINGS['figure_format'])
    else:
        fig.savefig(filepath, dpi=FIGURE_SETTINGS['dpi'],
                   format=FIGURE_SETTINGS['figure_format'])
    plt.close(fig)
    print(f"  ✓ Saved: {filename}")

def figure_1_sample_characteristics(df):
    """
    Figure 1: Sample Characteristics (4-panel)
    Shows distribution of key demographic variables.
    """
    print("\nCreating Figure 1: Sample Characteristics...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel A: Income Distribution
    income_order = ['Low', 'Medium', 'High']
    income_data = df['income_category'].value_counts().reindex(income_order)
    axes[0, 0].bar(income_order, income_data.values, 
                   color=[COLOR_PALETTE['primary'], COLOR_PALETTE['secondary'], 
                         COLOR_PALETTE['tertiary']])
    axes[0, 0].set_title('(A) Income Distribution', fontweight='bold', fontsize=14)
    axes[0, 0].set_ylabel('Number of Households', fontsize=12)
    axes[0, 0].set_xlabel('Income Category', fontsize=12)
    for i, v in enumerate(income_data.values):
        axes[0, 0].text(i, v + 50, f'{v:,}\n({v/len(df)*100:.1f}%)', 
                       ha='center', fontsize=10)
    
    # Panel B: Education Distribution
    edu_data = pd.Series({
        'No College': len(df[df['college_degree_plus'] == 0]),
        'College+': len(df[df['college_degree_plus'] == 1])
    })
    axes[0, 1].bar(edu_data.index, edu_data.values,
                   color=[COLOR_PALETTE['quaternary'], COLOR_PALETTE['tertiary']])
    axes[0, 1].set_title('(B) Education Level', fontweight='bold', fontsize=14)
    axes[0, 1].set_ylabel('Number of Households', fontsize=12)
    axes[0, 1].set_xlabel('Education Category', fontsize=12)
    for i, v in enumerate(edu_data.values):
        axes[0, 1].text(i, v + 100, f'{v:,}\n({v/len(df)*100:.1f}%)', 
                       ha='center', fontsize=10)
    
    # Panel C: Geographic Distribution
    region_data = df['region'].value_counts().head(6)
    axes[1, 0].barh(range(len(region_data)), region_data.values,
                    color=COLOR_PALETTE['secondary'])
    axes[1, 0].set_yticks(range(len(region_data)))
    axes[1, 0].set_yticklabels(region_data.index, fontsize=10)
    axes[1, 0].set_title('(C) Geographic Distribution', fontweight='bold', fontsize=14)
    axes[1, 0].set_xlabel('Number of Vehicles', fontsize=12)
    axes[1, 0].invert_yaxis()
    for i, v in enumerate(region_data.values):
        axes[1, 0].text(v + 50, i, f'{v:,} ({v/len(df)*100:.1f}%)', 
                       va='center', fontsize=10)
    
    # Panel D: Household Size Distribution
    hh_size_data = df['household_size'].value_counts().sort_index()
    axes[1, 1].plot(hh_size_data.index, hh_size_data.values, 
                   marker='o', linewidth=2, markersize=8,
                   color=COLOR_PALETTE['primary'])
    axes[1, 1].set_title('(D) Household Size Distribution', fontweight='bold', fontsize=14)
    axes[1, 1].set_xlabel('Number of Household Members', fontsize=12)
    axes[1, 1].set_ylabel('Number of Households', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'figure_01_sample_characteristics.png')

def figure_2_ev_adoption_patterns(df):
    """
    Figure 2: EV Adoption Patterns by Demographics (4-panel)
    """
    print("\nCreating Figure 2: EV Adoption Patterns...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel A: Adoption by Income
    income_order = ['Low', 'Medium', 'High']
    adoption_by_income = df.groupby('income_category')['is_ev'].mean().reindex(income_order) * 100
    axes[0, 0].bar(income_order, adoption_by_income.values,
                   color=[COLOR_PALETTE['ev'] if x > 10 else COLOR_PALETTE['non_ev'] 
                         for x in adoption_by_income.values])
    axes[0, 0].set_title('(A) EV Adoption by Income', fontweight='bold', fontsize=14)
    axes[0, 0].set_ylabel('EV Adoption Rate (%)', fontsize=12)
    axes[0, 0].set_xlabel('Income Category', fontsize=12)
    axes[0, 0].axhline(y=12.1, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Overall Rate')
    axes[0, 0].legend()
    for i, v in enumerate(adoption_by_income.values):
        axes[0, 0].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    # Panel B: Adoption by Education
    adoption_by_edu = df.groupby('college_degree_plus')['is_ev'].mean() * 100
    edu_labels = ['No College', 'College+']
    axes[0, 1].bar(edu_labels, adoption_by_edu.values,
                   color=[COLOR_PALETTE['non_ev'], COLOR_PALETTE['ev']])
    axes[0, 1].set_title('(B) EV Adoption by Education', fontweight='bold', fontsize=14)
    axes[0, 1].set_ylabel('EV Adoption Rate (%)', fontsize=12)
    axes[0, 1].set_xlabel('Education Level', fontsize=12)
    axes[0, 1].axhline(y=12.1, color='red', linestyle='--', linewidth=1, alpha=0.5)
    for i, v in enumerate(adoption_by_edu.values):
        axes[0, 1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    # Panel C: Adoption by Region
    adoption_by_region = df.groupby('region')['is_ev'].mean().sort_values(ascending=False).head(6) * 100
    axes[1, 0].barh(range(len(adoption_by_region)), adoption_by_region.values,
                    color=COLOR_PALETTE['tertiary'])
    axes[1, 0].set_yticks(range(len(adoption_by_region)))
    axes[1, 0].set_yticklabels(adoption_by_region.index, fontsize=10)
    axes[1, 0].set_title('(C) EV Adoption by Region', fontweight='bold', fontsize=14)
    axes[1, 0].set_xlabel('EV Adoption Rate (%)', fontsize=12)
    axes[1, 0].invert_yaxis()
    axes[1, 0].axvline(x=12.1, color='red', linestyle='--', linewidth=1, alpha=0.5)
    for i, v in enumerate(adoption_by_region.values):
        axes[1, 0].text(v + 0.3, i, f'{v:.1f}%', va='center', fontsize=10)
    
    # Panel D: Adoption by Mileage Category
    mileage_order = ['Low', 'Medium', 'High']
    adoption_by_mileage = df.groupby('mileage_category')['is_ev'].mean().reindex(mileage_order) * 100
    axes[1, 1].bar(mileage_order, adoption_by_mileage.values,
                   color=COLOR_PALETTE['quaternary'])
    axes[1, 1].set_title('(D) EV Adoption by Annual Mileage', fontweight='bold', fontsize=14)
    axes[1, 1].set_ylabel('EV Adoption Rate (%)', fontsize=12)
    axes[1, 1].set_xlabel('Mileage Category', fontsize=12)
    axes[1, 1].axhline(y=12.1, color='red', linestyle='--', linewidth=1, alpha=0.5)
    for i, v in enumerate(adoption_by_mileage.values):
        axes[1, 1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, 'figure_02_adoption_patterns.png')

def figure_3_target_distribution(df):
    """
    Figure 3: Target Variable Distribution (pie + bar)
    """
    print("\nCreating Figure 3: Target Variable Distribution...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Panel A: Pie chart
    ev_counts = df['is_ev'].value_counts()
    labels = ['Non-EV\n(Gasoline)', 'EV\n(BEV + PHEV)']
    colors = [COLOR_PALETTE['non_ev'], COLOR_PALETTE['ev']]
    explode = (0, 0.1)
    
    axes[0].pie(ev_counts.values, labels=labels, autopct='%1.1f%%',
               startangle=90, colors=colors, explode=explode,
               textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[0].set_title('(A) Overall Distribution', fontweight='bold', fontsize=14)
    
    # Panel B: Bar with counts
    axes[1].bar(['Non-EV', 'EV'], ev_counts.values, color=colors)
    axes[1].set_ylabel('Number of Vehicles', fontsize=12)
    axes[1].set_title('(B) Absolute Counts', fontweight='bold', fontsize=14)
    for i, v in enumerate(ev_counts.values):
        axes[1].text(i, v + 100, f'{v:,}', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, 'figure_03_target_distribution.png')

def figure_4_readiness_scores(df):
    """
    Figure 4: EV Adoption Readiness Metrics
    """
    print("\nCreating Figure 4: Adoption Readiness Scores...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel A: EV Experience Score Distribution
    axes[0, 0].hist(df['ev_experience_score'], bins=20, edgecolor='black',
                   color=COLOR_PALETTE['primary'], alpha=0.7)
    axes[0, 0].axvline(df['ev_experience_score'].mean(), color='red', 
                      linestyle='--', linewidth=2, label=f'Mean = {df["ev_experience_score"].mean():.2f}')
    axes[0, 0].set_title('(A) EV Experience Score', fontweight='bold', fontsize=14)
    axes[0, 0].set_xlabel('Score (0-2)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].legend()
    
    # Panel B: Charging Access Index Distribution
    axes[0, 1].hist(df['charging_access_index'], bins=20, edgecolor='black',
                   color=COLOR_PALETTE['secondary'], alpha=0.7)
    axes[0, 1].axvline(df['charging_access_index'].mean(), color='red',
                      linestyle='--', linewidth=2, label=f'Mean = {df["charging_access_index"].mean():.2f}')
    axes[0, 1].set_title('(B) Charging Access Index', fontweight='bold', fontsize=14)
    axes[0, 1].set_xlabel('Index (0-1)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].legend()
    
    # Panel C: Adoption Readiness Score Distribution
    axes[1, 0].hist(df['adoption_readiness_score'], bins=30, edgecolor='black',
                   color=COLOR_PALETTE['tertiary'], alpha=0.7)
    axes[1, 0].axvline(df['adoption_readiness_score'].mean(), color='red',
                      linestyle='--', linewidth=2, label=f'Mean = {df["adoption_readiness_score"].mean():.2f}')
    axes[1, 0].set_title('(C) Adoption Readiness Score', fontweight='bold', fontsize=14)
    axes[1, 0].set_xlabel('Score (0-10)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].legend()
    
    # Panel D: Readiness by EV Ownership
    readiness_comparison = df.groupby('is_ev')['adoption_readiness_score'].mean()
    axes[1, 1].bar(['Non-EV', 'EV'], readiness_comparison.values,
                  color=[COLOR_PALETTE['non_ev'], COLOR_PALETTE['ev']])
    axes[1, 1].set_title('(D) Readiness: Non-EV vs EV', fontweight='bold', fontsize=14)
    axes[1, 1].set_ylabel('Average Readiness Score', fontsize=12)
    for i, v in enumerate(readiness_comparison.values):
        axes[1, 1].text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    save_figure(fig, 'figure_04_readiness_scores.png')

def figure_5_vehicle_characteristics(df):
    """
    Figure 5: Vehicle Characteristics
    """
    print("\nCreating Figure 5: Vehicle Characteristics...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel A: Top 10 Vehicle Classes
    top_classes = df['veh_class_nrel'].value_counts().head(10)
    axes[0, 0].barh(range(len(top_classes)), top_classes.values,
                    color=COLOR_PALETTE['primary'])
    axes[0, 0].set_yticks(range(len(top_classes)))
    axes[0, 0].set_yticklabels(top_classes.index, fontsize=9)
    axes[0, 0].set_title('(A) Top 10 Vehicle Classes', fontweight='bold', fontsize=14)
    axes[0, 0].set_xlabel('Number of Vehicles', fontsize=12)
    axes[0, 0].invert_yaxis()
    
    # Panel B: Annual Mileage Distribution
    axes[0, 1].hist(df['annual_mileage'], bins=50, edgecolor='black',
                   color=COLOR_PALETTE['secondary'], alpha=0.7)
    axes[0, 1].axvline(df['annual_mileage'].mean(), color='red',
                      linestyle='--', linewidth=2, label=f'Mean = {df["annual_mileage"].mean():,.0f}')
    axes[0, 1].set_title('(B) Annual Mileage Distribution', fontweight='bold', fontsize=14)
    axes[0, 1].set_xlabel('Miles per Year', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].set_xlim(0, 30000)
    
    # Panel C: Vehicle Age Distribution
    axes[1, 0].hist(df['vehicle_age_approx'], bins=30, edgecolor='black',
                   color=COLOR_PALETTE['tertiary'], alpha=0.7)
    axes[1, 0].axvline(df['vehicle_age_approx'].mean(), color='red',
                      linestyle='--', linewidth=2, label=f'Mean = {df["vehicle_age_approx"].mean():.1f}')
    axes[1, 0].set_title('(C) Vehicle Age Distribution', fontweight='bold', fontsize=14)
    axes[1, 0].set_xlabel('Years', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].legend()
    
    # Panel D: Multi-Vehicle Households
    multi_veh = df['multi_vehicle_household'].value_counts()
    axes[1, 1].bar(['Single Vehicle', 'Multi-Vehicle'], multi_veh.values,
                  color=[COLOR_PALETTE['quaternary'], COLOR_PALETTE['primary']])
    axes[1, 1].set_title('(D) Multi-Vehicle Households', fontweight='bold', fontsize=14)
    axes[1, 1].set_ylabel('Number of Households', fontsize=12)
    for i, v in enumerate(multi_veh.values):
        axes[1, 1].text(i, v + 50, f'{v:,}\n({v/len(df)*100:.1f}%)', 
                       ha='center', fontsize=11)
    
    plt.tight_layout()
    save_figure(fig, 'figure_05_vehicle_characteristics.png')

def main():
    """Main function to generate all figures."""
    print("="*80)
    print("WEEK 2: EXPLORATORY DATA ANALYSIS - VISUALIZATIONS")
    print("="*80)
    
    # Load data
    df = load_data()
    
    # Create figures directory if it doesn't exist
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving figures to: {FIGURES_DIR}")
    
    # Generate all figures
    print("\n" + "="*80)
    print("GENERATING PUBLICATION-GRADE FIGURES (600 DPI)")
    print("="*80)
    
    figure_1_sample_characteristics(df)
    figure_2_ev_adoption_patterns(df)
    figure_3_target_distribution(df)
    figure_4_readiness_scores(df)
    figure_5_vehicle_characteristics(df)
    
    print("\n" + "="*80)
    print("✓ ALL FIGURES GENERATED")
    print("="*80)
    print(f"\nGenerated 5 figures in: {FIGURES_DIR}")
    print("\nNext step: Run statistical_analysis.py for hypothesis testing")

if __name__ == "__main__":
    main()