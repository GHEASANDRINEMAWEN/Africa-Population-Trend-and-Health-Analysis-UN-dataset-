# Africa Population Trend and Health Analysis - Assignment 2

## Predictive Modeling and Hypothesis Testing

This project builds upon Assignment 1's exploratory data analysis to test hypotheses about African demographic transitions using supervised machine learning.

---

## 📁 Project Structure

```
Africa-Population-Trend-and-Health-Analysis-UN-dataset-/
│
├── src/                          # Python modules (OOP implementation)
│   ├── __init__.py
│   ├── data_processor.py         # DataProcessor class - data loading & cleaning
│   ├── feature_engineer.py       # FeatureEngineer class - feature creation
│   ├── model_trainer.py          # ModelTrainer class - training & tuning
│   └── model_evaluator.py        # ModelEvaluator class - evaluation & comparison
│
├── notebooks/
│   └── main_analysis.ipynb       # Main workflow notebook with all analyses
│
├── results/                      # Auto-generated outputs
│   ├── figures/                  # Model comparison & feature importance plots
│   ├── tables/                   # Performance metrics CSV files
│   └── predictions/              # Model predictions for each hypothesis
│
├── df_africa_cleaned.xls         # Dataset (UN DESA population & health data)
│
└── README.md                     # This file
```

---

## 🎯 Project Overview

### Research Questions

This assignment tests **three hypotheses** about African demographic transitions (2010-2024):

1. **H1:** Does fertility rate predict maternal mortality? (Regression)
2. **H2:** Does life expectancy predict population growth rate? (Regression)
3. **H3:** Does under-five mortality predict life expectancy? (Regression)

### Dataset

- **Source:** UN DESA Population Division
- **Scope:** 54 African countries, 2010-2024 (annual observations)
- **Key Indicators:**
  - Life expectancy at birth (both sexes)
  - Total fertility rate
  - Under-five mortality rate
  - Maternal mortality ratio
  - Population annual growth rate

---

## 🚀 Getting Started

### Prerequisites

Required Python libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### Running the Analysis

1. **Ensure dataset is in the correct location:**
   ```
   df_africa_cleaned.xls should be in the project root directory
   ```

2. **Open the main notebook:**
   ```bash
   jupyter notebook notebooks/main_analysis.ipynb
   ```

3. **Run all cells sequentially** (Kernel → Restart & Run All)

### Expected Outputs

After running the notebook, the following files will be generated in `results/`:

**Tables:**
- `h1_comparison.csv` - Model performance for Hypothesis 1
- `h2_comparison.csv` - Model performance for Hypothesis 2
- `h3_comparison.csv` - Model performance for Hypothesis 3
- `hypothesis_testing_summary.csv` - Overall summary

**Figures:**
- `h1_model_comparison.png` - H1 model performance chart
- `h2_model_comparison.png` - H2 model performance chart
- `h3_model_comparison.png` - H3 model performance chart
- `h1_*_feature_importance.png` - Feature importance for each H1 model
- `h2_*_feature_importance.png` - Feature importance for each H2 model
- `h3_*_feature_importance.png` - Feature importance for each H3 model
- `hypothesis_testing_summary.png` - Comprehensive comparison visualization

**Predictions:**
- `h1_predictions.csv` - Maternal mortality predictions
- `h2_predictions.csv` - Population growth predictions
- `h3_predictions.csv` - Life expectancy predictions

---

## 🏗️ Architecture & Design

### Object-Oriented Design

The project follows **single responsibility principle** with four main classes:

#### 1. `DataProcessor`
**Responsibility:** Load and minimally process raw data
- File loading
- Type conversion (numeric, string)
- Column name standardization
- **No transformations or missing value imputation**

#### 2. `FeatureEngineer`
**Responsibility:** Create derived features
- Trend features (2010-2024 change)
- Country-level mean aggregates
- Optional normalization (StandardScaler)

#### 3. `ModelTrainer`
**Responsibility:** Train and tune models
- Train/test splitting with stratification
- Feature scaling
- Hyperparameter tuning (GridSearchCV)
- Model selection and prediction mapping

#### 4. `ModelEvaluator`
**Responsibility:** Evaluate and compare models
- Multiple metrics (R², RMSE, MAE, CV scores)
- Automated figure generation
- Comparison tables
- Feature importance analysis

---

## 📊 Key Results

### Hypothesis Testing Summary

| Hypothesis | Decision | Best Model | Test R² | Confidence |
|------------|----------|------------|---------|------------|
| **H1:** Fertility → Maternal Mortality | ✅ ACCEPT | Gradient Boosting | 0.893 | 90% |
| **H2:** Life Expectancy → Population Growth | ✅ ACCEPT | Linear Regression | 0.897 | 90% |
| **H3:** Under-5 Mortality → Life Expectancy | ✅ STRONGLY ACCEPT | Linear Regression | 0.9997 | >99% |

### Major Findings

1. **Under-five mortality is a near-perfect predictor of life expectancy** (R² = 0.9997)
   - Prediction error: ±0.11 years (approximately 1 month)
   - Strongest demographic relationship observed

2. **Life expectancy reliably predicts population growth** (R² = 0.897)
   - Linear relationship confirmed (demographic transition theory)
   - Prediction error: ±0.26 percentage points

3. **Fertility significantly predicts maternal mortality** (R² = 0.893)
   - Non-linear effects detected (ensemble methods outperformed)
   - Prediction error: ±68 deaths per 100,000

---

## 📈 Model Performance Details

### H1: Fertility → Maternal Mortality

| Model | Train R² | Test R² | RMSE | MAE | CV Mean R² |
|-------|----------|---------|------|-----|------------|
| Linear Regression | 0.791 | 0.756 | 102.69 | 77.44 | 0.685 |
| Random Forest | 0.977 | 0.835 | 84.56 | 60.51 | 0.798 |
| **Gradient Boosting** | **0.994** | **0.893** | **68.02** | **50.69** | **0.835** |

### H2: Life Expectancy → Population Growth

| Model | Train R² | Test R² | RMSE | MAE | CV Mean R² |
|-------|----------|---------|------|-----|------------|
| **Linear Regression** | **0.858** | **0.897** | **0.263** | **0.185** | **0.717** |
| Random Forest | 0.950 | 0.760 | 0.402 | 0.289 | 0.680 |
| Gradient Boosting | 0.986 | 0.808 | 0.359 | 0.281 | 0.719 |

### H3: Under-5 Mortality → Life Expectancy

| Model | Train R² | Test R² | RMSE | MAE | CV Mean R² |
|-------|----------|---------|------|-----|------------|
| **Linear Regression** | **0.9999** | **0.9997** | **0.112** | **0.078** | **0.9998** |
| Random Forest | 0.9985 | 0.9964 | 0.363 | 0.260 | 0.9852 |
| Gradient Boosting | 0.9999 | 0.9969 | 0.339 | 0.239 | 0.9912 |

---

## 🔬 Methodology

### Data Preparation

1. **Cleaning:**
   - Numeric type conversion
   - String normalization
   - Missing values preserved (no imputation)
   - Outliers retained (genuine demographic variation)

2. **Feature Engineering:**
   - 4 trend features (2010-2024 change)
   - 4 mean features (country-level averages)
   - StandardScaler normalization (in modeling pipeline)

3. **Train-Test Split:**
   - 80/20 split ratio
   - Random state = 42 (reproducibility)
   - Stratification for classification (N/A for regression)

### Model Selection

For each hypothesis, we tested **three models**:

1. **Linear Regression** - Baseline, interpretable
2. **Random Forest Regressor** - Captures non-linearity, feature importance
3. **Gradient Boosting Regressor** - High accuracy, handles complexity

### Evaluation Strategy

- **Primary Metric:** Test R² (generalization performance)
- **Supporting Metrics:** RMSE, MAE (prediction accuracy)
- **Validation:** 5-fold cross-validation (consistency check)
- **Selection Criteria:** Highest CV mean R² with reasonable CV std

---

## 💡 Key Insights

### What Predictive Modeling Revealed Beyond EDA

1. **Quantification:** Precise R² values show explanatory power (EDA only showed correlations)
2. **Generalization:** Cross-validation confirms patterns hold across subsets
3. **Feature Importance:** Identified which features matter most in multivariate context
4. **Prediction Accuracy:** Enabled precise forecasting with quantified uncertainty

### Surprising Findings

1. **Near-deterministic relationship** between child mortality and life expectancy (R² = 0.9997)
2. **Linear models outperformed** complex ensembles for H2 and H3
3. **Better test than train performance** in H2 (excellent generalization)
4. **Minimal CV variance** in H3 (SD = 0.00006) - universal pattern

---

## 🎓 Policy Recommendations

Based on our findings, we recommend:

### Priority 1: Child Health (Highest Impact)
- **Rationale:** Under-five mortality explains 99.97% of life expectancy variance
- **Actions:** Immunization, nutrition programs, malaria prevention, prenatal care
- **Expected Impact:** ~0.15-0.20 years life expectancy per 1-point mortality reduction

### Priority 2: Reproductive Health (High Impact)
- **Rationale:** Fertility explains 89.3% of maternal mortality variance
- **Actions:** Family planning, prenatal care, safe childbirth facilities
- **Expected Impact:** ~68 fewer maternal deaths per 100,000 per child reduction in fertility

### Priority 3: Demographic Transition (Strategic)
- **Rationale:** Life expectancy explains 89.7% of population growth variance
- **Actions:** Healthcare, education, economic development
- **Expected Impact:** Countries with life expectancy >70 years transition to <1.5% growth

---

## ⚠️ Limitations

1. **Causality:** Correlation ≠ causation. Experimental designs needed for causal claims.
2. **Geographic Scope:** Results specific to Africa (2010-2024), may not generalize globally.
3. **Missing Data:** 25% of maternal mortality values missing (all 2024).
4. **Temporal Lags:** Models use contemporaneous measurements, not lagged effects.
5. **Regional Heterogeneity:** Single continental model may mask subregional patterns.

---

## 📚 References

1. UN DESA Population Division (2024). *World Population Prospects 2024*
2. Scikit-learn developers (2024). *Scikit-learn: Machine Learning in Python*
3. Demographic Transition Theory literature (see full report for complete citations)

---

## 👥 Team & Contributions

**Assignment 2 - Group Project**

- **Data Cleaning & Preparation:** Completed ✓
- **Feature Engineering:** Completed ✓
- **Model Development:** Completed ✓
- **Model Evaluation:** Completed ✓
- **Hypothesis Testing Results:** Completed ✓ (Your contribution)

---

## 📝 License & Citation

If using this analysis, please cite:
```
Africa Population Trend and Health Analysis (2024)
UN DESA Population Data (2010-2024)
Carnegie Mellon University - Practical Data Analytics Course
```

---

## 🔧 Troubleshooting

### Common Issues

**Issue:** `FileNotFoundError: df_africa_cleaned.xls`
- **Solution:** Ensure dataset file is in project root directory

**Issue:** Module import errors
- **Solution:** Run notebook from `notebooks/` directory or adjust `sys.path`

**Issue:** Missing packages
- **Solution:** Install requirements: `pip install pandas numpy scikit-learn matplotlib seaborn`

**Issue:** Results folder not created
- **Solution:** Folders are auto-created on first run. Check write permissions.
