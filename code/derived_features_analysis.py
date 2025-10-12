#!/usr/bin/env python3
"""
Derived Features Analysis
Analyzes the dataset to suggest 5 most promising derived features for predicting y.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def load_and_analyze_data():
    """Load and analyze the dataset to understand feature characteristics."""
    print("="*80)
    print("DATASET ANALYSIS FOR DERIVED FEATURES")
    print("="*80)
    
    # Load data
    df = pd.read_csv("/Users/herve/Downloads/classif/data/train_processed.csv")
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution: {df['y'].value_counts().to_dict()}")
    print(f"Target rate: {df['y'].mean():.3f}")
    
    # Analyze numerical features
    numerical_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
    print(f"\nNumerical features: {numerical_features}")
    
    # Basic statistics
    print(f"\nNumerical feature statistics:")
    print(df[numerical_features].describe())
    
    # Analyze categorical features
    categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
    print(f"\nCategorical features: {categorical_features}")
    
    # Check for date-like features (none in this dataset)
    print(f"\nDate features analysis:")
    print("No explicit date features found in the dataset")
    print("However, 'pdays' represents days since last contact (temporal information)")
    
    return df, numerical_features, categorical_features

def suggest_derived_features(df, numerical_features, categorical_features):
    """Suggest 5 most promising derived features based on domain knowledge and data analysis."""
    print("\n" + "="*80)
    print("SUGGESTING 5 MOST PROMISING DERIVED FEATURES")
    print("="*80)
    
    # Create a copy for feature engineering
    df_features = df.copy()
    
    print("Creating derived features based on domain knowledge...")
    
    # 1. RISK PROFILE SCORE (Combines multiple risk indicators)
    print("\n1. RISK PROFILE SCORE")
    print("   Rationale: Combines multiple risk indicators that banks typically use")
    print("   - Default history, housing loan, personal loan, negative balance")
    
    df_features['risk_profile_score'] = (
        (df_features['default'] == 'yes').astype(int) * 3 +  # High weight for default
        (df_features['housing'] == 'yes').astype(int) * 1 +  # Medium weight for housing loan
        (df_features['loan'] == 'yes').astype(int) * 2 +     # High weight for personal loan
        (df_features['balance'] < 0).astype(int) * 2 +       # High weight for negative balance
        (df_features['balance'] == 0).astype(int) * 1        # Medium weight for zero balance
    )
    
    print(f"   Risk score distribution: {df_features['risk_profile_score'].value_counts().sort_index().to_dict()}")
    
    # 2. ENGAGEMENT INTENSITY (Measures customer engagement level)
    print("\n2. ENGAGEMENT INTENSITY")
    print("   Rationale: Customers who are more engaged are more likely to respond")
    print("   - Long call duration, multiple contacts, recent contact, previous success")
    
    df_features['engagement_intensity'] = (
        (df_features['duration'] > df_features['duration'].quantile(0.75)).astype(int) * 2 +  # Long calls
        (df_features['campaign'] > 1).astype(int) * 1 +                                        # Multiple contacts
        (df_features['pdays'] < 1000).astype(int) * 1 +                                       # Has been contacted before
        (df_features['previous'] > 0).astype(int) * 1 +                                       # Previous contact history
        (df_features['poutcome'] == 'success').astype(int) * 2                                # Previous success
    )
    
    print(f"   Engagement distribution: {df_features['engagement_intensity'].value_counts().sort_index().to_dict()}")
    
    # 3. FINANCIAL STABILITY INDEX (Age-adjusted financial health)
    print("\n3. FINANCIAL STABILITY INDEX")
    print("   Rationale: Financial stability relative to age is a strong predictor")
    print("   - Balance per age, account age, financial responsibility indicators")
    
    # Create age groups for better financial assessment
    df_features['age_group'] = pd.cut(df_features['age'], 
                                     bins=[0, 25, 35, 45, 55, 65, 100], 
                                     labels=[1, 2, 3, 4, 5, 6])
    
    df_features['financial_stability_index'] = (
        (df_features['balance'] / (df_features['age'] + 1)) * 1000 +  # Balance per age (scaled)
        (df_features['age'] > 30).astype(int) * 10 +                   # Mature age bonus
        (df_features['default'] == 'no').astype(int) * 20 +            # No default history
        (df_features['balance'] > 0).astype(int) * 15                  # Positive balance
    )
    
    print(f"   Financial stability range: {df_features['financial_stability_index'].min():.1f} to {df_features['financial_stability_index'].max():.1f}")
    
    # 4. CONTACT EFFICIENCY (Success rate of previous contacts)
    print("\n4. CONTACT EFFICIENCY")
    print("   Rationale: Past success rate is a strong predictor of future success")
    print("   - Previous success rate, contact frequency, recency")
    
    # Calculate success rate (previous successes / total previous contacts)
    df_features['contact_efficiency'] = np.where(
        df_features['previous'] > 0,
        (df_features['poutcome'] == 'success').astype(int) / (df_features['previous'] + 1),
        0  # No previous contact
    )
    
    # Add recency bonus (more recent contact = higher efficiency)
    df_features['contact_efficiency'] += np.where(
        df_features['pdays'] < 1000,
        (1000 - df_features['pdays']) / 1000,  # Recency bonus (0 to 1)
        0
    )
    
    print(f"   Contact efficiency range: {df_features['contact_efficiency'].min():.3f} to {df_features['contact_efficiency'].max():.3f}")
    
    # 5. LIFESTYLE RISK ASSESSMENT (Job and marital status combination)
    print("\n5. LIFESTYLE RISK ASSESSMENT")
    print("   Rationale: Job stability and marital status affect financial behavior")
    print("   - Job stability, marital status, education level combination")
    
    # Create job stability score
    job_stability = {
        'retired': 5, 'management': 4, 'technician': 4, 'admin.': 3,
        'services': 3, 'blue-collar': 2, 'self-employed': 2, 'entrepreneur': 2,
        'housemaid': 2, 'student': 1, 'unemployed': 0, 'unknown': 1
    }
    
    df_features['job_stability_score'] = df_features['job'].map(job_stability)
    
    # Create marital stability score
    marital_stability = {'married': 3, 'divorced': 1, 'single': 2}
    df_features['marital_stability_score'] = df_features['marital'].map(marital_stability)
    
    # Create education level score
    education_level = {'tertiary': 3, 'secondary': 2, 'primary': 1, 'unknown': 1}
    df_features['education_level_score'] = df_features['education'].map(education_level)
    
    # Combine into lifestyle risk assessment
    df_features['lifestyle_risk_assessment'] = (
        df_features['job_stability_score'] * 2 +      # Job stability (most important)
        df_features['marital_stability_score'] * 1 +  # Marital status
        df_features['education_level_score'] * 1      # Education level
    )
    
    print(f"   Lifestyle risk range: {df_features['lifestyle_risk_assessment'].min()} to {df_features['lifestyle_risk_assessment'].max()}")
    
    return df_features

def evaluate_derived_features(df_features):
    """Evaluate the predictive power of derived features."""
    print("\n" + "="*80)
    print("EVALUATING DERIVED FEATURES")
    print("="*80)
    
    # Prepare data for modeling
    target = df_features['y']
    
    # Select derived features
    derived_features = [
        'risk_profile_score',
        'engagement_intensity', 
        'financial_stability_index',
        'contact_efficiency',
        'lifestyle_risk_assessment'
    ]
    
    # Prepare features for correlation analysis
    features_df = df_features[derived_features]
    
    # Calculate correlation with target
    print("Correlation with target variable:")
    print("-" * 40)
    correlations = features_df.corrwith(target).sort_values(ascending=False)
    for feature, corr in correlations.items():
        print(f"{feature:<30}: {corr:.4f}")
    
    # Calculate mutual information (using our function)
    from simple_mi_function import mutual_info
    
    print(f"\nMutual Information with target:")
    print("-" * 40)
    mi_scores = {}
    for feature in derived_features:
        mi_score = mutual_info(df_features[feature], target, feature)
        mi_scores[feature] = mi_score
    
    # Sort by mutual information
    sorted_mi = sorted(mi_scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nRanked by Mutual Information:")
    print("-" * 40)
    for i, (feature, mi_score) in enumerate(sorted_mi, 1):
        print(f"{i}. {feature:<30}: {mi_score:.4f}")
    
    return sorted_mi, correlations

def create_visualization(df_features, sorted_mi):
    """Create visualizations for the derived features."""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    # Create subplots for each derived feature
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    derived_features = [item[0] for item in sorted_mi]
    
    for i, feature in enumerate(derived_features):
        if i < 5:  # Only plot the 5 derived features
            ax = axes[i]
            
            # Create box plot showing distribution by target
            df_features.boxplot(column=feature, by='y', ax=ax)
            ax.set_title(f'{feature}\nMI: {sorted_mi[i][1]:.4f}')
            ax.set_xlabel('Target (y)')
            ax.set_ylabel('Feature Value')
    
    # Remove empty subplot
    axes[5].remove()
    
    plt.suptitle('Distribution of Top 5 Derived Features by Target Variable', fontsize=16)
    plt.tight_layout()
    plt.savefig('/Users/herve/Downloads/classif/derived_features_analysis.png', 
                dpi=300, bbox_inches='tight')
    print("Visualization saved as: derived_features_analysis.png")
    plt.show()

def create_feature_importance_plot(sorted_mi):
    """Create a bar plot of feature importance."""
    print("\nCreating feature importance plot...")
    
    features = [item[0] for item in sorted_mi]
    mi_scores = [item[1] for item in sorted_mi]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(features)), mi_scores, color='lightgreen', alpha=0.7)
    
    plt.yticks(range(len(features)), features)
    plt.xlabel('Mutual Information Score')
    plt.title('Top 5 Derived Features by Mutual Information\n(Higher = More Predictive)', 
              fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, mi_scores)):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{score:.4f}', ha='left', va='center', fontsize=10)
    
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('/Users/herve/Downloads/classif/derived_features_importance.png', 
                dpi=300, bbox_inches='tight')
    print("Feature importance plot saved as: derived_features_importance.png")
    plt.show()

def main():
    """Main function to run the derived features analysis."""
    # Load and analyze data
    df, numerical_features, categorical_features = load_and_analyze_data()
    
    # Suggest derived features
    df_features = suggest_derived_features(df, numerical_features, categorical_features)
    
    # Evaluate derived features
    sorted_mi, correlations = evaluate_derived_features(df_features)
    
    # Create visualizations
    create_visualization(df_features, sorted_mi)
    # create_feature_importance_plot(sorted_mi)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF TOP 5 DERIVED FEATURES")
    print("="*80)
    
    print("1. RISK PROFILE SCORE")
    print("   - Combines default, loans, and balance indicators")
    print("   - Higher score = higher risk = less likely to respond")
    
    print("\n2. ENGAGEMENT INTENSITY") 
    print("   - Measures customer engagement level")
    print("   - Higher score = more engaged = more likely to respond")
    
    print("\n3. FINANCIAL STABILITY INDEX")
    print("   - Age-adjusted financial health measure")
    print("   - Higher score = more stable = more likely to respond")
    
    print("\n4. CONTACT EFFICIENCY")
    print("   - Past success rate and contact recency")
    print("   - Higher score = better past performance = more likely to respond")
    
    print("\n5. LIFESTYLE RISK ASSESSMENT")
    print("   - Job stability, marital status, education combination")
    print("   - Higher score = more stable lifestyle = more likely to respond")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    main()
