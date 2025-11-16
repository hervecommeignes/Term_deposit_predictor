#!/usr/bin/env python3
"""
New Base Model
Optimized XGBoost model with risk_score feature and tuned hyperparameters.
This is the new baseline model for further feature engineering.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def create_engagement_intensity(df):
    """
    Create engagement intensity feature based on customer interaction patterns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with engagement_intensity feature added
    """
    df['engagement_intensity'] = (
        (df['duration'] > df['duration'].quantile(0.75)).astype(int) * 2 +
        (df['campaign'] > 1).astype(int) * 1 +
        (df['pdays'] < 1000).astype(int) * 1 +
        (df['previous'] > 0).astype(int) * 1 +
        (df['poutcome'] == 'success').astype(int) * 2
    )
    return df

def create_risk_score(df):
    """
    Create comprehensive risk score based on multiple risk factors.
    
    Risk factors:
    - Default: 3 points (high impact)
    - Housing loan: 2 points (financial burden)
    - Personal loan: 2 points (financial burden)
    - Negative balance: 3 points (financial distress)
    - Age > 65: 1 point (elderly risk)
    - Previous failure: 2 points (historical performance)
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with risk_score feature added
    """
    df['risk_score'] = (
        (df['default'] == 'yes').astype(int) * 3 +
        (df['housing'] == 'yes').astype(int) * 2 +
        (df['loan'] == 'yes').astype(int) * 2 +
        (df['balance'] < 0).astype(int) * 3 +
        (df['age'] > 65).astype(int) * 1 +
        (df['poutcome'] == 'failure').astype(int) * 2
    )
    return df

def prepare_data(train_path, test_path):
    """
    Load and prepare data for the new base model.
    
    Args:
        train_path (str): Path to training data
        test_path (str): Path to test data
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
    """
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Combine for feature engineering
    df = pd.concat([train_df, test_df], ignore_index=True)
    print(f"Combined data shape: {df.shape}")
    
    # Create engineered features
    df = create_engagement_intensity(df)
    df = create_risk_score(df)
    
    # Define features for new base model
    feature_names = ['duration', 'age', 'balance', 'engagement_intensity', 'risk_score']
    
    # Prepare features and target
    X = df[feature_names]
    y = df['y']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Features: {feature_names}")
    
    return X_train, X_test, y_train, y_test, feature_names

def create_new_base_model(feature_names: list):
    """
    Create the new base model with optimized hyperparameters.
    
    Model configuration:
    - Feature interactions DISABLED (each feature used independently)
    - Monotonic constraints ENABLED (approximate enforcement)
    
    Args:
        feature_names (list): List of feature names (required)
                             Used to create interaction constraints and monotonic constraints.
    
    Returns:
        xgb.XGBClassifier: Optimized XGBoost model
    """
    # Base hyperparameters
    params = {
        'n_estimators': 200,        # More trees for better learning
        'max_depth': 8,             # Deeper trees for complex patterns
        'learning_rate': 0.05,      # Slower learning for better generalization
        'subsample': 0.8,           # Prevent overfitting
        'colsample_bytree': 0.8,    # Feature sampling
        'reg_alpha': 0.1,           # L1 regularization
        'reg_lambda': 1.0,          # L2 regularization
        'random_state': 42,
        'verbosity': 0,
        'eval_metric': 'logloss',
        'tree_method': 'hist'       # Required for monotonic constraints support
    }
    
    # Disable feature interactions by constraining each feature to its own group
    # This prevents any two features from appearing together in the same tree path
    # When using DataFrames, XGBoost expects feature names, not indices
    # Format: [['feature0'], ['feature1'], ...] means each feature can only interact with itself
    if feature_names is None:
        raise ValueError("feature_names must be provided")
    params['interaction_constraints'] = [[name] for name in feature_names]
    
    # Set monotonic constraints
    # NOTE: XGBoost's monotonic constraints are APPROXIMATE and guide tree building.
    # They do NOT guarantee perfect monotonicity, especially in sparse data regions.
    # The constraints help enforce general monotonic trends but may allow small violations.
    # Format: list of integers, one per feature in feature order
    # 1 = monotonic increasing, -1 = monotonic decreasing, 0 = no constraint
    # Feature order: ['duration', 'age', 'balance', 'engagement_intensity', 'risk_score']
    # Based on domain knowledge:
    # - duration: increasing (longer calls → more conversion)
    # - age: no constraint (could have optimal age range)
    # - balance: increasing (more money → more likely to convert)
    # - engagement_intensity: increasing (higher engagement → more conversion)
    # - risk_score: decreasing (higher risk → less likely to convert)
    if feature_names is not None:
        monotonic_map = {
            'duration': 1,              # Increasing: longer calls → more conversion
            'age': 0,                   # No constraint: optimal age range
            'balance': 1,               # Increasing: more money → more likely
            'engagement_intensity': 1,  # Increasing: higher engagement → more conversion
            'risk_score': -1            # Decreasing: higher risk → less conversion
        }
        # Create monotonic constraints list in feature order
        params['monotonic_constraints'] = [monotonic_map.get(fname, 0) for fname in feature_names]
    
    model = xgb.XGBClassifier(**params)
    
    return model

def train_and_evaluate_model(X_train, X_test, y_train, y_test, feature_names):
    """
    Train the new base model and evaluate its performance.
    
    The model has:
    - Feature interactions DISABLED (each feature used independently)
    - Monotonic constraints ENABLED (approximate enforcement)
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        y_train (pd.Series): Training target
        y_test (pd.Series): Test target
        feature_names (list): List of feature names
        
    Returns:
        dict: Model performance metrics and trained model
    """
    print("\n" + "="*80)
    print("NEW BASE MODEL TRAINING AND EVALUATION")
    print("="*80)
    
    # Create and train model
    model = create_new_base_model(feature_names=feature_names)
    
    # Display model constraints
    monotonic = model.get_params().get('monotonic_constraints')
    interaction = model.get_params().get('interaction_constraints')
    print(f"Training new base model...")
    print(f"  Feature interactions: DISABLED")
    print(f"  Monotonic constraints: {monotonic}")
    print(f"  Interaction constraints: {len(interaction)} groups (one per feature)")
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Find optimal threshold
    from sklearn.metrics import precision_recall_curve
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-8)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    y_pred_optimal = (y_pred_proba > optimal_threshold).astype(int)
    precision_optimal = precision_score(y_test, y_pred_optimal)
    recall_optimal = recall_score(y_test, y_pred_optimal)
    
    print(f"\nModel Performance:")
    print(f"  ROC-AUC: {auc:.4f}")
    print(f"  Feature interactions: DISABLED - each feature used independently")
    print(f"  Monotonic constraints: ENABLED (approximate enforcement)")
    print(f"  Precision (0.5 threshold): {precision:.4f}")
    print(f"  Recall (0.5 threshold): {recall:.4f}")
    print(f"  F1-Score (0.5 threshold): {f1:.4f}")
    print(f"  Optimal Threshold: {optimal_threshold:.3f}")
    print(f"  Precision (optimal): {precision_optimal:.4f}")
    print(f"  Recall (optimal): {recall_optimal:.4f}")
    print(f"  F1-Score (optimal): {optimal_f1:.4f}")
    
    # Cross-validation
    print(f"\nCross-validation (3-fold):")
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc', n_jobs=-1)
    print(f"  Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance:")
    for _, row in feature_importance.iterrows():
        print(f"  {row['feature']:<20}: {row['importance']:.4f}")
    
    # Discrete feature analysis (risk_score and engagement_intensity)
    # Both are discrete integer features that can be analyzed by grouping
    print(f"\nDiscrete Feature Analysis:")
    
    # Risk score analysis
    print(f"\n  Risk Score Analysis:")
    risk_analysis = pd.concat([X_test, y_test], axis=1).groupby('risk_score')['y'].agg(['count', 'sum', 'mean'])
    for risk_level in sorted(risk_analysis.index):
        count = risk_analysis.loc[risk_level, 'count']
        target_rate = risk_analysis.loc[risk_level, 'mean']
        print(f"    Risk {risk_level}: {count:6d} samples, {target_rate:.3f} target rate")
    
    # Engagement intensity analysis
    print(f"\n  Engagement Intensity Analysis:")
    engagement_analysis = pd.concat([X_test, y_test], axis=1).groupby('engagement_intensity')['y'].agg(['count', 'sum', 'mean'])
    for eng_level in sorted(engagement_analysis.index):
        count = engagement_analysis.loc[eng_level, 'count']
        target_rate = engagement_analysis.loc[eng_level, 'mean']
        print(f"    Engagement {eng_level}: {count:6d} samples, {target_rate:.3f} target rate")
    
    return {
        'model': model,
        'auc': auc,
        'precision': precision_optimal,
        'recall': recall_optimal,
        'f1': optimal_f1,
        'threshold': optimal_threshold,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'feature_importance': feature_importance
    }

def create_visualization(results: dict, feature_names: list) -> None:
    """
    Create visualization of model performance and feature importance.
    
    Args:
        results (dict): Model performance results containing:
            - 'feature_importance': DataFrame with 'feature' and 'importance' columns
            - 'auc': float, ROC-AUC score
            - 'precision': float, precision score
            - 'recall': float, recall score
            - 'f1': float, F1 score
        feature_names (list): List of feature names (for validation)
    """
    try:
        print("\nCreating visualizations...")
        
        # Validate inputs
        if 'feature_importance' not in results:
            raise ValueError("Results dictionary must contain 'feature_importance' key")
        
        importance_df = results['feature_importance']
        
        # Ensure importance_df is a DataFrame with required columns
        if not isinstance(importance_df, pd.DataFrame):
            raise TypeError("feature_importance must be a pandas DataFrame")
        
        if 'feature' not in importance_df.columns or 'importance' not in importance_df.columns:
            raise ValueError("feature_importance DataFrame must contain 'feature' and 'importance' columns")
        
        # Ensure importance values are numeric
        importance_df = importance_df.copy()
        importance_df['importance'] = pd.to_numeric(importance_df['importance'], errors='coerce')
        
        # Sort by importance for better visualization
        importance_df = importance_df.sort_values('importance', ascending=True)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Feature importance plot
        bars = ax1.barh(range(len(importance_df)), importance_df['importance'].values, 
                       color='lightgreen', alpha=0.7)
        ax1.set_yticks(range(len(importance_df)))
        ax1.set_yticklabels(importance_df['feature'].values)
        ax1.set_xlabel('Feature Importance')
        ax1.set_title('New Base Model - Feature Importance', fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, importance in zip(bars, importance_df['importance'].values):
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{importance:.3f}', ha='left', va='center', fontsize=10)
        
        # Performance metrics
        metrics = ['ROC-AUC', 'Precision', 'Recall', 'F1-Score']
        values = [
            float(results['auc']), 
            float(results['precision']), 
            float(results['recall']), 
            float(results['f1'])
        ]
        
        bars2 = ax2.bar(metrics, values, color='lightblue', alpha=0.7)
        ax2.set_ylabel('Score')
        ax2.set_title('New Base Model - Performance Metrics', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars2, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        output_path = '/Users/herve/Downloads/classif/new_base_model_performance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close figure to free memory
        print(f"Visualization saved as: new_base_model_performance.png")
        
    except Exception as e:
        print(f"\nError creating visualization: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise

def save_model(model, feature_names: list, results: dict) -> None:
    """
    Save the trained model and metadata.
    
    Args:
        model: Trained XGBoost model
        feature_names (list): List of feature names
        results (dict): Model performance results containing:
            - 'auc': float, ROC-AUC score
            - 'precision': float, precision score
            - 'recall': float, recall score
            - 'f1': float, F1 score
            - 'threshold': float, optimal threshold
            - 'feature_importance': DataFrame with 'feature' and 'importance' columns
    """
    import joblib
    import json
    
    # Save model
    model_path = '/Users/herve/Downloads/classif/new_base_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Convert feature importance DataFrame to dict with native Python types
    feature_importance_dict = results['feature_importance'].copy()
    # Convert importance column to native Python float
    feature_importance_dict['importance'] = feature_importance_dict['importance'].astype(float)
    feature_importance_records = feature_importance_dict.to_dict('records')
    
    # Ensure all values in feature_importance records are native Python types
    for record in feature_importance_records:
        record['importance'] = float(record['importance'])
    
    # Get model hyperparameters for metadata
    model_params = model.get_params()
    
    # Save metadata with all values converted to native Python types
    metadata = {
        'feature_names': feature_names,
        'hyperparameters': {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'monotonic_constraints': model_params.get('monotonic_constraints'),
            'interaction_constraints': 'disabled' if model_params.get('interaction_constraints') else 'enabled'
        },
        'performance': {
            'auc': float(results['auc']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1': float(results['f1']),
            'threshold': float(results['threshold'])
        },
        'feature_importance': feature_importance_records
    }
    
    metadata_path = '/Users/herve/Downloads/classif/new_base_model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

def main():
    """
    Main function to create and evaluate the new base model.
    
    Creates a model with:
    - Feature interactions DISABLED (each feature used independently)
    - Monotonic constraints ENABLED (approximate enforcement)
      * duration: increasing
      * balance: increasing
      * engagement_intensity: increasing
      * risk_score: decreasing
      * age: no constraint
    """
    print("="*80)
    print("NEW BASE MODEL CREATION")
    print("="*80)
    
    # Data paths
    train_path = "/Users/herve/Downloads/classif/data/train_processed.csv"
    test_path = "/Users/herve/Downloads/classif/data/test_processed.csv"
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_names = prepare_data(train_path, test_path)
    
    # Train and evaluate model WITHOUT feature interactions and WITH monotonic constraints
    results = train_and_evaluate_model(
        X_train, X_test, y_train, y_test, feature_names
    )
    
    # Create visualizations
    create_visualization(results, feature_names)
    
    # Save model
    save_model(results['model'], feature_names, results)
    
    print("\n" + "="*80)
    print("NEW BASE MODEL CREATION COMPLETED!")
    print("="*80)
    print(f"✅ Model ROC-AUC: {results['auc']:.4f}")
    print(f"✅ Features: {len(feature_names)} ({', '.join(feature_names)})")
    print(f"✅ Derived features preserved: engagement_intensity, risk_score")
    print(f"✅ Feature interactions: DISABLED (each feature used independently)")
    print(f"✅ Monotonic constraints: ENABLED (approximate enforcement)")
    print(f"   - duration: increasing")
    print(f"   - balance: increasing")
    print(f"   - engagement_intensity: increasing")
    print(f"   - risk_score: decreasing")
    print(f"   - age: no constraint")
    
    return results

if __name__ == "__main__":
    results = main()
