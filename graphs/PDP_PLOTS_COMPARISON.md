# Partial Dependency Plots Comparison

## Overview

Two types of Partial Dependency Plot (PDP) visualizations are generated for the model analysis. Both show how individual features affect model predictions, but differ in the level of detail and additional context provided.

---

## `individual_pdp_plots.png`

**Purpose:** Basic individual analysis of each feature's partial dependence.

### Key Features:
- ✅ PDP curve for each feature
- ✅ Mean and median reference lines (red/orange dashed)
- ✅ Grid layout (2×3) with separate subplot per feature
- ✅ Simple, clean visualization

### When to Use:
- Quick overview of feature effects
- Comparing relative feature importance visually
- Presentations needing simple, clear plots

### What It Shows:
- How each feature individually affects predictions
- Reference points (mean/median) for feature values
- Overall trend patterns per feature

---

## `detailed_pdp_analysis.png`

**Purpose:** Deep statistical analysis with data distribution context.

### Key Features:
- ✅ PDP curve (left y-axis) + Data density histogram (right y-axis)
- ✅ Dual y-axes showing both model behavior and data distribution
- ✅ Statistical summary box: Std, Range, Mean, Median, Max Slope
- ✅ Maximum slope line (green) indicating fastest change region
- ✅ Mean, median, and max slope reference lines

### When to Use:
- Understanding data reliability (where data is sparse vs dense)
- Identifying regions of rapid change (max slope)
- Assessing extrapolation risk (PDP in sparse data regions)
- Statistical analysis and model validation

### What It Shows:
- Partial dependence + data density simultaneously
- Feature statistics (std, range, mean, median)
- Rate of change (slope) analysis
- Data concentration (critical for reliability assessment)

---

## Key Differences Summary

| Aspect | `individual_pdp_plots.png` | `detailed_pdp_analysis.png` |
|--------|---------------------------|------------------------------|
| **Data Distribution** | Mean/median lines only | Full histogram overlay |
| **Y-Axes** | Single (PDP only) | Dual (PDP + density) |
| **Statistics** | Mean, median | Std, range, mean, median, max slope |
| **Slope Analysis** | ❌ No | ✅ Yes (max slope line) |
| **Reliability Context** | ❌ No | ✅ Yes (histogram shows data concentration) |
| **Complexity** | Simple | Detailed |

---

## Visual Example

### `individual_pdp_plots.png`
```
Duration Plot:
- Blue PDP line
- Red line: Mean
- Orange line: Median
```

### `detailed_pdp_analysis.png`
```
Duration Plot:
- Blue PDP line (left axis)
- Gray histogram (right axis)
- Red line: Mean
- Orange line: Median  
- Green line: Max slope point
- Text box: Std, Range, Max Slope
```

---

## Interpretation Guide

### For `individual_pdp_plots.png`:
- **Question:** "What's the general trend?"
- **Answer:** Look at the PDP curve direction and shape

### For `detailed_pdp_analysis.png`:
- **Question:** "Is this prediction reliable here?"
- **Answer:** Check if the histogram (gray bars) shows data density at that point
- **Question:** "Where does the effect change fastest?"
- **Answer:** Find the green max slope line

---

## Recommendation

- **Start with** `individual_pdp_plots.png` for quick insights
- **Deep dive with** `detailed_pdp_analysis.png` for reliability assessment and statistical analysis
- **Use both** for comprehensive understanding: trends from individual plots, reliability from detailed analysis

---

## Note on Non-Monotonic Behavior

Both plots show non-monotonic patterns despite monotonic constraints being set. This is **expected** because:

- XGBoost's monotonic constraints are **approximate**, not hard constraints
- They guide tree building but don't guarantee perfect monotonicity
- Non-monotonicity is especially visible in sparse data regions

The detailed plot helps identify whether non-monotonic behavior occurs in data-dense regions (more concerning) or sparse regions (less reliable due to extrapolation).

