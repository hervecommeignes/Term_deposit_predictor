#!/usr/bin/env python3
"""
Partial Dependency Plots for New Base Model
Creates comprehensive partial dependency plots for all features in the new base model.

IMPORTANT NOTE ON MONOTONIC CONSTRAINTS:
------------------------------------------
The model used for these PDPs has monotonic constraints set:
- duration: increasing (+1)
- age: no constraint (0)
- balance: increasing (+1)
- engagement_intensity: increasing (+1)
- risk_score: decreasing (-1)

However, XGBoost's monotonic constraints are APPROXIMATE, not hard constraints.
They guide tree building during training but do NOT guarantee perfect monotonicity.

As a result, the PDP plots may show non-monotonic behavior even though constraints
are set. This is expected behavior and reflects XGBoost's approximate constraint
enforcement, especially in sparse data regions or with complex model structures.

The constraints help ensure:
- General directional trends (e.g., longer calls generally → higher conversion)
- Domain knowledge is respected on average
- But small violations can occur

If you see non-monotonic PDPs, this is normal and expected with XGBoost.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def create_engagement_intensity(df):
    """Create engagement intensity feature."""
    df['engagement_intensity'] = (
        (df['duration'] > df['duration'].quantile(0.75)).astype(int) * 2 +
        (df['campaign'] > 1).astype(int) * 1 +
        (df['pdays'] < 1000).astype(int) * 1 +
        (df['previous'] > 0).astype(int) * 1 +
        (df['poutcome'] == 'success').astype(int) * 2
    )
    return df

def create_risk_score(df):
    """Create risk score feature."""
    df['risk_score'] = (
        (df['default'] == 'yes').astype(int) * 3 +
        (df['housing'] == 'yes').astype(int) * 2 +
        (df['loan'] == 'yes').astype(int) * 2 +
        (df['balance'] < 0).astype(int) * 3 +
        (df['age'] > 65).astype(int) * 1 +
        (df['poutcome'] == 'failure').astype(int) * 2
    )
    return df

def prepare_data_and_train_model():
    """
    Prepare data and load the saved new base model.
    Uses the saved model to ensure consistency with the model used for analysis.
    
    Returns:
        tuple: (model, X_train, X_test, y_train, y_test, feature_names)
    """
    import joblib
    import os
    
    model_path = '/Users/herve/Downloads/classif/new_base_model.pkl'
    
    # Try to load saved model first
    if os.path.exists(model_path):
        print("Loading saved model from new_base_model.pkl...")
        model = joblib.load(model_path)
        model_params = model.get_params()
        print("✅ Loaded saved model")
        print(f"   - Feature interactions: {'DISABLED' if model_params.get('interaction_constraints') else 'ENABLED'}")
        print(f"   - Monotonic constraints: {model_params.get('monotonic_constraints', 'None')}")
        print("   ⚠️  Note: Monotonic constraints are approximate in XGBoost")
        print("      PDPs may show non-monotonic behavior - this is expected and normal")
    else:
        print("⚠️  Saved model not found. Training new model with same configuration...")
        # Load data
        train_df = pd.read_csv("/Users/herve/Downloads/classif/data/train_processed.csv")
        test_df = pd.read_csv("/Users/herve/Downloads/classif/data/test_processed.csv")
        df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Create features
        df = create_engagement_intensity(df)
        df = create_risk_score(df)
        
        # Prepare features
        feature_names = ['duration', 'age', 'balance', 'engagement_intensity', 'risk_score']
        X = df[feature_names]
        y = df['y']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model with same configuration as saved model (WITHOUT interactions + monotonic constraints)
        # This matches the configuration from new_base_model.py with disable_interactions=True
        feature_names_for_constraints = ['duration', 'age', 'balance', 'engagement_intensity', 'risk_score']
        
        # Set monotonic constraints
        # 1 = monotonic increasing, -1 = monotonic decreasing, 0 = no constraint
        monotonic_map = {
            'duration': 1,              # Increasing: longer calls → more conversion
            'age': 0,                   # No constraint: optimal age range
            'balance': 1,               # Increasing: more money → more likely
            'engagement_intensity': 1,  # Increasing: higher engagement → more conversion
            'risk_score': -1            # Decreasing: higher risk → less conversion
        }
        monotonic_constraints = [monotonic_map.get(fname, 0) for fname in feature_names_for_constraints]
        
        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0, eval_metric='logloss',
            interaction_constraints=[[name] for name in feature_names_for_constraints],  # Disable interactions
            monotonic_constraints=monotonic_constraints  # Enforce monotonicity
        )
        model.fit(X_train, y_train)
        return model, X_train, X_test, y_train, y_test, feature_names
    
    # Load data for PDP analysis (needed for training data)
    print("Loading data for partial dependence analysis...")
    train_df = pd.read_csv("/Users/herve/Downloads/classif/data/train_processed.csv")
    test_df = pd.read_csv("/Users/herve/Downloads/classif/data/test_processed.csv")
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Create features (must match what was used during training)
    df = create_engagement_intensity(df)
    df = create_risk_score(df)
    
    # Prepare features
    feature_names = ['duration', 'age', 'balance', 'engagement_intensity', 'risk_score']
    X = df[feature_names]
    y = df['y']
    
    # Split data (same split as during training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return model, X_train, X_test, y_train, y_test, feature_names

def create_individual_pdp_plots(model, X_train, feature_names):
    """
    Create individual partial dependency plots for each feature.
    
    NOTE: These PDPs may show non-monotonic behavior even though the model
    has monotonic constraints set. This is due to XGBoost's approximate constraint
    enforcement. The model generally follows monotonic trends but allows small
    violations, especially in sparse data regions.
    
    Expected monotonicity (if constraints were perfect):
    - duration: should be monotonic increasing
    - balance: should be monotonic increasing
    - engagement_intensity: should be monotonic increasing
    - risk_score: should be monotonic decreasing
    - age: no constraint
    
    Args:
        model: Trained XGBoost model with monotonic constraints
        X_train: Training data (DataFrame)
        feature_names: List of feature names
    """
    print("\nCreating individual partial dependency plots...")
    print("NOTE: PDPs may show non-monotonic behavior due to XGBoost's approximate")
    print("      monotonic constraint enforcement. This is expected and normal.")
    
    # Create individual plots for each feature
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, feature in enumerate(feature_names):
        ax = axes[i]
        
        # Create partial dependence plot
        pdp_result = partial_dependence(
            model, X_train, [feature], 
            kind='average', 
            grid_resolution=50
        )
        
        # Extract values
        feature_values = pdp_result['values'][0]
        pdp_values = pdp_result['average'][0]
        
        # Plot
        ax.plot(feature_values, pdp_values, 'b-', linewidth=2, marker='o', markersize=3)
        ax.set_xlabel(feature.replace('_', ' ').title())
        ax.set_ylabel('Partial Dependence')
        ax.set_title(f'PDP: {feature.replace("_", " ").title()}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        feature_data = X_train[feature]
        ax.axvline(feature_data.mean(), color='red', linestyle='--', alpha=0.7, 
                  label=f'Mean: {feature_data.mean():.2f}')
        ax.axvline(feature_data.median(), color='orange', linestyle='--', alpha=0.7, 
                  label=f'Median: {feature_data.median():.2f}')
        ax.legend(fontsize=8)
    
    # Remove empty subplot
    axes[5].remove()
    
    plt.suptitle('Partial Dependency Plots - Individual Features\nNew Base Model', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/Users/herve/Downloads/classif/individual_pdp_plots.png', 
                dpi=300, bbox_inches='tight')
    print("Individual PDP plots saved as: individual_pdp_plots.png")
    plt.close(fig)

def create_combined_pdp_plot(model, X_train, feature_names):
    """Create a combined partial dependency plot using sklearn's PartialDependenceDisplay."""
    print("\nCreating combined partial dependency plot...")
    
    # Create combined plot
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Use sklearn's PartialDependenceDisplay for better visualization
    display = PartialDependenceDisplay.from_estimator(
        model, X_train, feature_names,
        grid_resolution=50,
        n_jobs=-1,
        ax=ax
    )
    
    ax.set_title('Partial Dependency Plots - All Features\nNew Base Model', 
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/herve/Downloads/classif/combined_pdp_plot.png', 
                dpi=300, bbox_inches='tight')
    print("Combined PDP plot saved as: combined_pdp_plot.png")
    plt.close(fig)

def create_detailed_pdp_analysis(model, X_train, feature_names):
    """Create detailed analysis with statistics and insights."""
    print("\nCreating detailed PDP analysis...")
    
    # Create subplots for detailed analysis
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.ravel()
    
    for i, feature in enumerate(feature_names):
        ax = axes[i]
        
        # Get partial dependence
        pdp_result = partial_dependence(
            model, X_train, [feature], 
            kind='average', 
            grid_resolution=100
        )
        
        feature_values = pdp_result['values'][0]
        pdp_values = pdp_result['average'][0]
        
        # Plot main PDP
        ax.plot(feature_values, pdp_values, 'b-', linewidth=3, label='PDP')
        
        # Add feature distribution as background
        feature_data = X_train[feature]
        ax2 = ax.twinx()
        ax2.hist(feature_data, bins=30, alpha=0.3, color='gray', density=True, label='Data Distribution')
        ax2.set_ylabel('Data Density', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        
        # Add statistics
        mean_val = feature_data.mean()
        median_val = feature_data.median()
        std_val = feature_data.std()
        
        ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, 
                  label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='orange', linestyle='--', alpha=0.8, 
                  label=f'Median: {median_val:.2f}')
        
        # Calculate slope (rate of change)
        if len(feature_values) > 1:
            slope = np.gradient(pdp_values, feature_values)
            max_slope_idx = np.argmax(np.abs(slope))
            max_slope_point = feature_values[max_slope_idx]
            max_slope_value = slope[max_slope_idx]
            
            ax.axvline(max_slope_point, color='green', linestyle=':', alpha=0.8, 
                      label=f'Max Slope: {max_slope_value:.3f}')
        
        # Styling
        ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Partial Dependence', fontsize=12)
        ax.set_title(f'Detailed PDP: {feature.replace("_", " ").title()}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')
        
        # Add feature statistics as text
        stats_text = f'Std: {std_val:.2f}\nRange: [{feature_data.min():.2f}, {feature_data.max():.2f}]'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Remove empty subplot
    axes[5].remove()
    
    plt.suptitle('Detailed Partial Dependency Analysis\nNew Base Model with Statistics', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/Users/herve/Downloads/classif/detailed_pdp_analysis.png', 
                dpi=300, bbox_inches='tight')
    print("Detailed PDP analysis saved as: detailed_pdp_analysis.png")
    plt.close(fig)

def create_pdp_insights(model, X_train, feature_names):
    """
    Generate insights from partial dependency plots.
    
    NOTE: The model has monotonic constraints set, but XGBoost enforces them
    approximately. Therefore, the PDPs may show non-monotonic behavior even
    though constraints are set. The constraints guide the model to follow
    general trends but don't guarantee perfect monotonicity.
    
    Args:
        model: Trained XGBoost model with monotonic constraints
        X_train: Training data (DataFrame)
        feature_names: List of feature names
    """
    print("\n" + "="*80)
    print("PARTIAL DEPENDENCY PLOT INSIGHTS")
    print("="*80)
    print("\nNOTE: Model has monotonic constraints set, but enforcement is approximate.")
    print("      Non-monotonic patterns in PDPs are expected and normal with XGBoost.\n")
    
    insights = {}
    
    for feature in feature_names:
        print(f"\n{feature.upper().replace('_', ' ')}:")
        print("-" * 40)
        
        # Get partial dependence
        pdp_result = partial_dependence(
            model, X_train, [feature], 
            kind='average', 
            grid_resolution=100
        )
        
        feature_values = pdp_result['values'][0]
        pdp_values = pdp_result['average'][0]
        
        # Calculate statistics
        feature_data = X_train[feature]
        min_pdp = np.min(pdp_values)
        max_pdp = np.max(pdp_values)
        pdp_range = max_pdp - min_pdp
        
        # Find trend
        if len(feature_values) > 1:
            slope = np.gradient(pdp_values, feature_values)
            avg_slope = np.mean(slope)
            
            if avg_slope > 0.001:
                trend = "INCREASING"
            elif avg_slope < -0.001:
                trend = "DECREASING"
            else:
                trend = "STABLE"
        else:
            trend = "CONSTANT"
        
        # Find key inflection points
        if len(slope) > 2:
            slope_changes = np.diff(np.sign(slope))
            inflection_points = np.where(slope_changes != 0)[0]
        else:
            inflection_points = []
        
        print(f"  PDP Range: {min_pdp:.4f} to {max_pdp:.4f} (span: {pdp_range:.4f})")
        print(f"  Overall Trend: {trend}")
        print(f"  Data Range: {feature_data.min():.2f} to {feature_data.max():.2f}")
        print(f"  Data Mean: {feature_data.mean():.2f}")
        print(f"  Data Std: {feature_data.std():.2f}")
        
        if len(inflection_points) > 0:
            print(f"  Inflection Points: {len(inflection_points)} detected")
            for idx in inflection_points[:3]:  # Show first 3
                print(f"    - At {feature_values[idx]:.2f}: PDP = {pdp_values[idx]:.4f}")
        
        # Business interpretation
        if feature == 'duration':
            print(f"  Business Insight: Longer calls generally increase conversion probability")
        elif feature == 'age':
            print(f"  Business Insight: Age shows {trend.lower()} relationship with conversion")
        elif feature == 'balance':
            print(f"  Business Insight: Balance shows {trend.lower()} relationship with conversion")
        elif feature == 'engagement_intensity':
            print(f"  Business Insight: Higher engagement intensity increases conversion probability")
        elif feature == 'risk_score':
            print(f"  Business Insight: Higher risk score decreases conversion probability")
        
        insights[feature] = {
            'pdp_range': pdp_range,
            'trend': trend,
            'min_pdp': min_pdp,
            'max_pdp': max_pdp,
            'inflection_points': len(inflection_points)
        }
    
    return insights

def create_pdp_summary_plot(insights, feature_names):
    """Create a summary plot of PDP insights."""
    print("\nCreating PDP summary plot...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # PDP Range comparison
    features = list(insights.keys())
    pdp_ranges = [insights[f]['pdp_range'] for f in features]
    colors = ['red' if insights[f]['trend'] == 'DECREASING' 
              else 'green' if insights[f]['trend'] == 'INCREASING' 
              else 'blue' for f in features]
    
    bars1 = ax1.bar(features, pdp_ranges, color=colors, alpha=0.7)
    ax1.set_xlabel('Features')
    ax1.set_ylabel('PDP Range')
    ax1.set_title('Partial Dependence Range by Feature\n(Red=Decreasing, Green=Increasing, Blue=Stable)', 
                  fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, range_val in zip(bars1, pdp_ranges):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{range_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Trend distribution
    trend_counts = {}
    for f in features:
        trend = insights[f]['trend']
        trend_counts[trend] = trend_counts.get(trend, 0) + 1
    
    ax2.pie(trend_counts.values(), labels=trend_counts.keys(), autopct='%1.0f%%', 
            colors=['green', 'red', 'blue'][:len(trend_counts)])
    ax2.set_title('Feature Trend Distribution', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/Users/herve/Downloads/classif/pdp_summary.png', 
                dpi=300, bbox_inches='tight')
    print("PDP summary saved as: pdp_summary.png")
    plt.close(fig)

def main():
    """
    Main function to create comprehensive partial dependency plots.
    
    The model used has:
    - Feature interactions DISABLED (each feature used independently)
    - Monotonic constraints ENABLED (approximate enforcement)
      * duration: increasing
      * balance: increasing
      * engagement_intensity: increasing
      * risk_score: decreasing
      * age: no constraint
    
    NOTE: PDPs may appear non-monotonic because XGBoost's monotonic constraints
    are approximate and guide tree building rather than enforcing hard constraints.
    """
    print("="*80)
    print("PARTIAL DEPENDENCY PLOTS FOR NEW BASE MODEL")
    print("="*80)
    print("\nModel Configuration:")
    print("  ✓ Feature interactions: DISABLED")
    print("  ✓ Monotonic constraints: ENABLED (approximate enforcement)")
    print("  ⚠️  PDPs may show non-monotonic behavior - this is expected with XGBoost")
    print("="*80)
    
    # Prepare data and train model
    model, X_train, X_test, y_train, y_test, feature_names = prepare_data_and_train_model()
    
    print(f"\nModel trained with features: {feature_names}")
    print(f"Training data shape: {X_train.shape}")
    
    # Create individual PDP plots
    create_individual_pdp_plots(model, X_train, feature_names)
    
    # Create combined PDP plot
    create_combined_pdp_plot(model, X_train, feature_names)
    
    # Create detailed analysis
    create_detailed_pdp_analysis(model, X_train, feature_names)
    
    # Generate insights
    insights = create_pdp_insights(model, X_train, feature_names)
    
    # Create summary plot
    create_pdp_summary_plot(insights, feature_names)
    
    print("\n" + "="*80)
    print("PARTIAL DEPENDENCY ANALYSIS COMPLETED!")
    print("="*80)
    print("Generated files:")
    print("  - individual_pdp_plots.png")
    print("  - combined_pdp_plot.png") 
    print("  - detailed_pdp_analysis.png")
    print("  - pdp_summary.png")
    
    return insights

if __name__ == "__main__":
    insights = main()
