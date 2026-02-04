"""
Exploratory Data Analysis Script
================================
Analyzes both Taiwan Bankruptcy and Ames Housing datasets
Generates visualizations saved to plots/ directory

Usage: python eda_analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data file paths (relative to project root or absolute)
DATA_PATH_CLASSIFICATION = '/Users/belinda/Desktop/MachineLearning_Courseowork/data.csv'
DATA_PATH_REGRESSION = '/Users/belinda/Desktop/MachineLearning_Courseowork/ames.csv'

# Output directory for plots
PLOTS_DIR = Path('plots')
PLOTS_DIR.mkdir(exist_ok=True)

# Plot styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10

# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def print_section_header(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)


def analyze_classification_data(filepath, target_col='Bankrupt?'):
    """
    Analyze Taiwan bankruptcy classification dataset
    
    Args:
        filepath: Path to data.csv
        target_col: Name of target column
    
    Returns:
        DataFrame with analysis results
    """
    print_section_header("TAIWAN BANKRUPTCY PREDICTION DATASET (CLASSIFICATION)")
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"\n📊 Dataset Dimensions: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Target variable analysis
    print(f"\n🎯 Target Variable: '{target_col}'")
    class_counts = df[target_col].value_counts()
    print(f"   Class 0 (Non-bankrupt): {class_counts[0]:,} ({class_counts[0]/len(df)*100:.2f}%)")
    print(f"   Class 1 (Bankrupt):     {class_counts[1]:,} ({class_counts[1]/len(df)*100:.2f}%)")
    print(f"   Imbalance Ratio: {class_counts[1]/class_counts[0]:.4f} (1:{class_counts[0]/class_counts[1]:.1f})")
    
    # Data types
    print(f"\n📋 Feature Types:")
    print(f"   Numeric:     {df.select_dtypes(include=[np.number]).shape[1]}")
    print(f"   Categorical: {df.select_dtypes(include=['object']).shape[1]}")
    
    # Missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"\n⚠️  Missing Values: {missing_count:,} total")
        missing_cols = df.columns[df.isnull().any()].tolist()
        print(f"   Affected columns: {missing_cols[:10]}")
    else:
        print(f"\n✅ Missing Values: None (0)")
    
    # Duplicates
    dup_count = df.duplicated().sum()
    print(f"\n🔍 Duplicate Rows: {dup_count}")
    
    # Feature statistics
    numeric_df = df.select_dtypes(include=[np.number])
    print(f"\n📈 Numeric Feature Statistics:")
    print(f"   Zero variance features: {(numeric_df.var() == 0).sum()}")
    print(f"   Features with variance < 0.01: {(numeric_df.var() < 0.01).sum()}")
    
    # Outliers (using IQR method)
    outlier_features = []
    for col in numeric_df.columns:
        Q1, Q3 = numeric_df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outliers = ((numeric_df[col] < (Q1 - 1.5*IQR)) | 
                   (numeric_df[col] > (Q3 + 1.5*IQR))).sum()
        if outliers > 0:
            outlier_features.append((col, outliers))
    
    print(f"   Features with outliers (IQR method): {len(outlier_features)}")
    if len(outlier_features) > 0:
        top_outliers = sorted(outlier_features, key=lambda x: x[1], reverse=True)[:5]
        print(f"   Top 5 features by outlier count:")
        for feat, count in top_outliers:
            print(f"      - {feat}: {count} ({count/len(df)*100:.1f}%)")
    
    return df


def analyze_regression_data(filepath, target_col='price'):
    """
    Analyze Ames housing regression dataset
    
    Args:
        filepath: Path to ames.csv
        target_col: Name of target column
    
    Returns:
        DataFrame with analysis results
    """
    print_section_header("AMES HOUSING PRICE PREDICTION DATASET (REGRESSION)")
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"\n📊 Dataset Dimensions: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Target variable analysis
    print(f"\n🎯 Target Variable: '{target_col}'")
    print(f"   Mean:     ${df[target_col].mean():,.2f}")
    print(f"   Median:   ${df[target_col].median():,.2f}")
    print(f"   Std Dev:  ${df[target_col].std():,.2f}")
    print(f"   Range:    ${df[target_col].min():,.2f} - ${df[target_col].max():,.2f}")
    print(f"   Skewness: {df[target_col].skew():.3f} (right-skewed; log transform recommended)")
    print(f"   Kurtosis: {df[target_col].kurtosis():.3f}")
    
    # Data types
    print(f"\n📋 Feature Types:")
    print(f"   Numeric:     {df.select_dtypes(include=[np.number]).shape[1]}")
    print(f"   Categorical: {df.select_dtypes(include=['object']).shape[1]}")
    
    # Missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Count': missing[missing > 0],
        'Percentage': missing_pct[missing > 0]
    }).sort_values('Percentage', ascending=False)
    
    if len(missing_df) > 0:
        print(f"\n⚠️  Missing Values: {len(missing_df)} columns affected")
        print(f"   Top 10 columns with missing data:")
        print(missing_df.head(10).to_string())
    else:
        print(f"\n✅ Missing Values: None (0)")
    
    # Duplicates
    dup_count = df.duplicated().sum()
    print(f"\n🔍 Duplicate Rows: {dup_count}")
    
    # Numeric feature statistics
    numeric_df = df.select_dtypes(include=[np.number])
    print(f"\n📈 Numeric Feature Statistics:")
    print(f"   Zero variance features: {(numeric_df.var() == 0).sum()}")
    print(f"   Features with variance < 1: {(numeric_df.var() < 1).sum()}")
    
    # Categorical feature analysis
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        print(f"\n🏷️  Categorical Features:")
        high_cardinality = []
        for col in cat_cols:
            n_unique = df[col].nunique()
            if n_unique > 20:
                high_cardinality.append((col, n_unique))
        
        print(f"   High cardinality (>20 categories): {len(high_cardinality)} features")
        if high_cardinality:
            for col, n in sorted(high_cardinality, key=lambda x: x[1], reverse=True)[:5]:
                print(f"      - {col}: {n} unique values")
    
    return df


def create_classification_plots(df, target_col='Bankrupt?'):
    """Create visualization for classification dataset"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Taiwan Bankruptcy Dataset - Exploratory Analysis', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Plot 1: Class distribution
    class_counts = df[target_col].value_counts()
    colors = ['#3498db', '#e74c3c']
    
    axes[0].bar(range(len(class_counts)), class_counts.values, color=colors, 
                edgecolor='black', linewidth=1.5, alpha=0.8)
    axes[0].set_xticks(range(len(class_counts)))
    axes[0].set_xticklabels(['Non-Bankrupt (0)', 'Bankrupt (1)'])
    axes[0].set_ylabel('Number of Companies', fontsize=12, fontweight='bold')
    axes[0].set_title('Class Distribution', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, (count, pct) in enumerate(zip(class_counts.values, 
                                         class_counts.values / len(df) * 100)):
        axes[0].text(i, count + len(df)*0.01, f'{count:,}\n({pct:.2f}%)',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add imbalance ratio annotation
    imbalance = class_counts[1] / class_counts[0]
    axes[0].text(0.5, 0.95, f'Imbalance Ratio: {imbalance:.4f} (1:{1/imbalance:.1f})',
                transform=axes[0].transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6),
                fontsize=10, fontweight='bold')
    
    # Plot 2: Top correlated features with target
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        correlations = df[numeric_cols].corr()[target_col].sort_values(ascending=False)
        top_10 = correlations[1:11]  # Top 10 excluding target
        
        colors_corr = ['#27ae60' if x > 0 else '#e74c3c' for x in top_10]
        top_10.plot(kind='barh', ax=axes[1], color=colors_corr, 
                    edgecolor='black', linewidth=1)
        
        axes[1].set_xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
        axes[1].set_title('Top 10 Features Correlated with Bankruptcy', 
                         fontsize=13, fontweight='bold')
        axes[1].axvline(x=0, color='black', linestyle='-', linewidth=1.5)
        axes[1].grid(axis='x', alpha=0.3)
        
        # Truncate long feature names for readability
        labels = [label.get_text()[:40] + '...' if len(label.get_text()) > 40 
                 else label.get_text() for label in axes[1].get_yticklabels()]
        axes[1].set_yticklabels(labels, fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    output_path = PLOTS_DIR / 'bankruptcy_exploratory_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Plot saved: {output_path}")
    plt.close()


def create_regression_plots(df, target_col='price'):
    """Create visualization for regression dataset"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Ames Housing Dataset - Exploratory Analysis', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Plot 1: Price distribution with statistics
    axes[0].hist(df[target_col], bins=60, color='#3498db', alpha=0.7, 
                edgecolor='black', linewidth=0.8)
    
    # Add vertical lines for mean and median
    mean_val = df[target_col].mean()
    median_val = df[target_col].median()
    
    axes[0].axvline(mean_val, color='red', linestyle='--', linewidth=2.5, 
                   label=f'Mean: ${mean_val:,.0f}')
    axes[0].axvline(median_val, color='green', linestyle='--', linewidth=2.5, 
                   label=f'Median: ${median_val:,.0f}')
    
    axes[0].set_xlabel('Sale Price ($)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Sale Price Distribution', fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add statistics box
    stats_text = f'Skewness: {df[target_col].skew():.3f}\nKurtosis: {df[target_col].kurtosis():.3f}'
    axes[0].text(0.98, 0.97, stats_text, transform=axes[0].transAxes,
                ha='right', va='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                fontsize=9, fontweight='bold')
    
    # Plot 2: Top correlated features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        correlations = df[numeric_cols].corr()[target_col].sort_values(ascending=False)
        top_10 = correlations[1:11]  # Top 10 excluding target
        
        colors_corr = ['#27ae60' if x > 0 else '#e74c3c' for x in top_10]
        top_10.plot(kind='barh', ax=axes[1], color=colors_corr, 
                    edgecolor='black', linewidth=1)
        
        axes[1].set_xlabel('Correlation Coefficient', fontsize=12, fontweight='bold')
        axes[1].set_title('Top 10 Features Correlated with Price', 
                         fontsize=13, fontweight='bold')
        axes[1].axvline(x=0, color='black', linestyle='-', linewidth=1.5)
        axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = PLOTS_DIR / 'housing_exploratory_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Plot saved: {output_path}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "🔬 EXPLORATORY DATA ANALYSIS ".center(80, "="))
    print(f"\n📁 Output Directory: {PLOTS_DIR.absolute()}\n")
    
    # Analyze classification dataset
    df_classification = analyze_classification_data(DATA_PATH_CLASSIFICATION)
    create_classification_plots(df_classification)
    
    # Analyze regression dataset
    df_regression = analyze_regression_data(DATA_PATH_REGRESSION)
    create_regression_plots(df_regression)
    
    # Summary
    print_section_header("ANALYSIS COMPLETE")
    print(f"\n✅ All visualizations saved to: {PLOTS_DIR.absolute()}/")
    print("\n📊 Generated files:")
    print(f"   1. bankruptcy_exploratory_analysis.png")
    print(f"   2. housing_exploratory_analysis.png")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()