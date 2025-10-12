#!/usr/bin/env python3
"""
Threshold Analysis for New Base Model
Creates precision-recall curves and threshold analysis to recommend optimal threshold.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
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
    """Prepare data and train the new base model."""
    print("Loading data and training model...")
    
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
    
    # Train model
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbosity=0
    )
    model.fit(X_train, y_train)
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    return model, X_test, y_test, y_pred_proba, feature_names

def analyze_thresholds(y_test, y_pred_proba):
    """Analyze precision and recall at different thresholds."""
    print("\nAnalyzing thresholds...")
    
    # Generate thresholds from 0.01 to 0.99
    thresholds = np.arange(0.01, 1.0, 0.01)
    
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for threshold in thresholds:
        y_pred = (y_pred_proba > threshold).astype(int)
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    
    return thresholds, precision_scores, recall_scores, f1_scores

def find_optimal_thresholds(thresholds, precision_scores, recall_scores, f1_scores):
    """Find optimal thresholds for different criteria."""
    print("\nFinding optimal thresholds...")
    
    # F1-score optimal
    f1_optimal_idx = np.argmax(f1_scores)
    f1_optimal_threshold = thresholds[f1_optimal_idx]
    f1_optimal_precision = precision_scores[f1_optimal_idx]
    f1_optimal_recall = recall_scores[f1_optimal_idx]
    f1_optimal_f1 = f1_scores[f1_optimal_idx]
    
    # Precision-Recall balance (closest to equal precision and recall)
    pr_diff = np.abs(np.array(precision_scores) - np.array(recall_scores))
    pr_balance_idx = np.argmin(pr_diff)
    pr_balance_threshold = thresholds[pr_balance_idx]
    pr_balance_precision = precision_scores[pr_balance_idx]
    pr_balance_recall = recall_scores[pr_balance_idx]
    pr_balance_f1 = f1_scores[pr_balance_idx]
    
    # High precision (precision > 0.7)
    high_precision_mask = np.array(precision_scores) > 0.7
    if np.any(high_precision_mask):
        high_precision_indices = np.where(high_precision_mask)[0]
        high_precision_f1_scores = np.array(f1_scores)[high_precision_indices]
        high_precision_best_idx = high_precision_indices[np.argmax(high_precision_f1_scores)]
        high_precision_threshold = thresholds[high_precision_best_idx]
        high_precision_precision = precision_scores[high_precision_best_idx]
        high_precision_recall = recall_scores[high_precision_best_idx]
        high_precision_f1 = f1_scores[high_precision_best_idx]
    else:
        high_precision_threshold = None
        high_precision_precision = None
        high_precision_recall = None
        high_precision_f1 = None
    
    # High recall (recall > 0.7)
    high_recall_mask = np.array(recall_scores) > 0.7
    if np.any(high_recall_mask):
        high_recall_indices = np.where(high_recall_mask)[0]
        high_recall_f1_scores = np.array(f1_scores)[high_recall_indices]
        high_recall_best_idx = high_recall_indices[np.argmax(high_recall_f1_scores)]
        high_recall_threshold = thresholds[high_recall_best_idx]
        high_recall_precision = precision_scores[high_recall_best_idx]
        high_recall_recall = recall_scores[high_recall_best_idx]
        high_recall_f1 = f1_scores[high_recall_best_idx]
    else:
        high_recall_threshold = None
        high_recall_precision = None
        high_recall_recall = None
        high_recall_f1 = None
    
    return {
        'f1_optimal': (f1_optimal_threshold, f1_optimal_precision, f1_optimal_recall, f1_optimal_f1),
        'pr_balance': (pr_balance_threshold, pr_balance_precision, pr_balance_recall, pr_balance_f1),
        'high_precision': (high_precision_threshold, high_precision_precision, high_precision_recall, high_precision_f1),
        'high_recall': (high_recall_threshold, high_recall_precision, high_recall_recall, high_recall_f1)
    }

def create_threshold_visualization(thresholds, precision_scores, recall_scores, f1_scores, optimal_thresholds):
    """Create comprehensive threshold analysis visualization."""
    print("\nCreating threshold analysis visualization...")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Precision, Recall, F1 vs Threshold
    ax1.plot(thresholds, precision_scores, 'b-', label='Precision', linewidth=2)
    ax1.plot(thresholds, recall_scores, 'r-', label='Recall', linewidth=2)
    ax1.plot(thresholds, f1_scores, 'g-', label='F1-Score', linewidth=2)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Precision, Recall, and F1-Score vs Threshold', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Add optimal threshold markers
    f1_thresh, f1_prec, f1_rec, f1_f1 = optimal_thresholds['f1_optimal']
    ax1.axvline(x=f1_thresh, color='g', linestyle='--', alpha=0.7, label=f'F1 Optimal: {f1_thresh:.3f}')
    ax1.legend()
    
    # 2. Precision vs Recall curve
    ax2.plot(recall_scores, precision_scores, 'purple', linewidth=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # Mark optimal points
    ax2.plot(f1_rec, f1_prec, 'go', markersize=8, label=f'F1 Optimal: P={f1_prec:.3f}, R={f1_rec:.3f}')
    
    pr_thresh, pr_prec, pr_rec, pr_f1 = optimal_thresholds['pr_balance']
    ax2.plot(pr_rec, pr_prec, 'ro', markersize=8, label=f'P-R Balance: P={pr_prec:.3f}, R={pr_rec:.3f}')
    ax2.legend()
    
    # 3. F1-Score vs Threshold (zoomed)
    ax3.plot(thresholds, f1_scores, 'g-', linewidth=2)
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('F1-Score')
    ax3.set_title('F1-Score vs Threshold (Detailed View)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=f1_thresh, color='g', linestyle='--', alpha=0.7)
    ax3.axhline(y=f1_f1, color='g', linestyle='--', alpha=0.7)
    ax3.text(f1_thresh, f1_f1, f'Optimal: {f1_thresh:.3f}\nF1: {f1_f1:.3f}', 
             ha='left', va='bottom', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 4. Threshold recommendations table
    ax4.axis('off')
    
    # Create recommendation table
    recommendations = []
    recommendations.append(['Criterion', 'Threshold', 'Precision', 'Recall', 'F1-Score'])
    recommendations.append(['F1 Optimal', f'{f1_thresh:.3f}', f'{f1_prec:.3f}', f'{f1_rec:.3f}', f'{f1_f1:.3f}'])
    recommendations.append(['P-R Balance', f'{pr_thresh:.3f}', f'{pr_prec:.3f}', f'{pr_rec:.3f}', f'{pr_f1:.3f}'])
    
    if optimal_thresholds['high_precision'][0] is not None:
        hp_thresh, hp_prec, hp_rec, hp_f1 = optimal_thresholds['high_precision']
        recommendations.append(['High Precision', f'{hp_thresh:.3f}', f'{hp_prec:.3f}', f'{hp_rec:.3f}', f'{hp_f1:.3f}'])
    
    if optimal_thresholds['high_recall'][0] is not None:
        hr_thresh, hr_prec, hr_rec, hr_f1 = optimal_thresholds['high_recall']
        recommendations.append(['High Recall', f'{hr_thresh:.3f}', f'{hr_prec:.3f}', f'{hr_rec:.3f}', f'{hr_f1:.3f}'])
    
    # Create table
    table = ax4.table(cellText=recommendations[1:], colLabels=recommendations[0], 
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(recommendations[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Threshold Recommendations', fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('/Users/herve/Downloads/classif/threshold_analysis.png', 
                dpi=300, bbox_inches='tight')
    print("Visualization saved as: threshold_analysis.png")
    plt.show()

def print_threshold_analysis(optimal_thresholds):
    """Print detailed threshold analysis and recommendations."""
    print("\n" + "="*80)
    print("THRESHOLD ANALYSIS RESULTS")
    print("="*80)
    
    f1_thresh, f1_prec, f1_rec, f1_f1 = optimal_thresholds['f1_optimal']
    pr_thresh, pr_prec, pr_rec, pr_f1 = optimal_thresholds['pr_balance']
    
    print(f"\n1. F1-SCORE OPTIMAL THRESHOLD:")
    print(f"   Threshold: {f1_thresh:.3f}")
    print(f"   Precision: {f1_prec:.3f}")
    print(f"   Recall:    {f1_rec:.3f}")
    print(f"   F1-Score:  {f1_f1:.3f}")
    
    print(f"\n2. PRECISION-RECALL BALANCE THRESHOLD:")
    print(f"   Threshold: {pr_thresh:.3f}")
    print(f"   Precision: {pr_prec:.3f}")
    print(f"   Recall:    {pr_rec:.3f}")
    print(f"   F1-Score:  {pr_f1:.3f}")
    
    if optimal_thresholds['high_precision'][0] is not None:
        hp_thresh, hp_prec, hp_rec, hp_f1 = optimal_thresholds['high_precision']
        print(f"\n3. HIGH PRECISION THRESHOLD (P > 0.7):")
        print(f"   Threshold: {hp_thresh:.3f}")
        print(f"   Precision: {hp_prec:.3f}")
        print(f"   Recall:    {hp_rec:.3f}")
        print(f"   F1-Score:  {hp_f1:.3f}")
    
    if optimal_thresholds['high_recall'][0] is not None:
        hr_thresh, hr_prec, hr_rec, hr_f1 = optimal_thresholds['high_recall']
        print(f"\n4. HIGH RECALL THRESHOLD (R > 0.7):")
        print(f"   Threshold: {hr_thresh:.3f}")
        print(f"   Precision: {hr_prec:.3f}")
        print(f"   Recall:    {hr_rec:.3f}")
        print(f"   F1-Score:  {hr_f1:.3f}")
    
    print(f"\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    print(f"\n🎯 RECOMMENDED THRESHOLD: {f1_thresh:.3f}")
    print(f"   Rationale: Maximizes F1-Score for balanced precision and recall")
    print(f"   Performance: Precision={f1_prec:.3f}, Recall={f1_rec:.3f}, F1={f1_f1:.3f}")
    
    print(f"\n📊 BUSINESS CONTEXT RECOMMENDATIONS:")
    print(f"   • For balanced performance: Use {f1_thresh:.3f} (F1 optimal)")
    print(f"   • For precision-focused: Use {pr_thresh:.3f} (P-R balance)")
    if optimal_thresholds['high_precision'][0] is not None:
        print(f"   • For high precision: Use {optimal_thresholds['high_precision'][0]:.3f}")
    if optimal_thresholds['high_recall'][0] is not None:
        print(f"   • For high recall: Use {optimal_thresholds['high_recall'][0]:.3f}")
    
    print(f"\n💡 INTERPRETATION:")
    print(f"   • Threshold {f1_thresh:.3f} means: Predict positive if probability > {f1_thresh:.1%}")
    print(f"   • This balances false positives and false negatives")
    print(f"   • Consider business costs of false positives vs false negatives")

def main():
    """Main function to run threshold analysis."""
    print("="*80)
    print("THRESHOLD ANALYSIS FOR NEW BASE MODEL")
    print("="*80)
    
    # Prepare data and train model
    model, X_test, y_test, y_pred_proba, feature_names = prepare_data_and_train_model()
    
    # Analyze thresholds
    thresholds, precision_scores, recall_scores, f1_scores = analyze_thresholds(y_test, y_pred_proba)
    
    # Find optimal thresholds
    optimal_thresholds = find_optimal_thresholds(thresholds, precision_scores, recall_scores, f1_scores)
    
    # Create visualization
    create_threshold_visualization(thresholds, precision_scores, recall_scores, f1_scores, optimal_thresholds)
    
    # Print analysis
    print_threshold_analysis(optimal_thresholds)
    
    print("\n" + "="*80)
    print("THRESHOLD ANALYSIS COMPLETED!")
    print("="*80)
    
    return optimal_thresholds

if __name__ == "__main__":
    optimal_thresholds = main()
