# Africa Population Trend and Health Analysis

## Assignment 2: Predictive Modeling and Hypothesis Testing

This project builds upon exploratory data analysis (Assignment 1) to test hypotheses and make predictions using supervised machine learning models on UN DESA population and health indicators for African countries (2010-2024).

---

## 📋 Project Overview

### Research Questions
This project tests three key hypotheses about African demographic health patterns:

1. **H1:** Does fertility predict maternal mortality?
2. **H2:** Does life expectancy predict population growth?
3. **H3:** Does under-five mortality predict life expectancy?

### Key Findings
- **All three hypotheses were ACCEPTED** based on strong model performance
- Best performing models achieved R² scores of 0.83, 0.77, and 0.997 respectively
- Feature importance analysis revealed that healthcare quality indicators dominate predictions

For detailed results, see [`HYPOTHESIS_TESTING_SUMMARY.md`](HYPOTHESIS_TESTING_SUMMARY.md)

---

## 📁 Project Structure

```
project_directory/
│
├── src/                          # Source code modules (OOP implementation)
│   ├── __init__.py              # Package initialization
│   ├── data_processor.py        # DataProcessor class
│   ├── feature_engineer.py      # FeatureEngineer class
│   ├── model_trainer.py         # ModelTrainer class
│   └── model_evaluator.py       # ModelEvaluator class
│
├── notebooks/
│   └── main_analysis.ipynb      # Main workflow notebook
│
├── df_africa_cleaned.xls        # Dataset (UN DESA indicators 2010-2024)
├── HYPOTHESIS_TESTING_SUMMARY.md # Detailed hypothesis testing results
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook
- Required libraries (see requirements.txt)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/GHEASANDRINEMAWEN/Africa-Population-Trend-and-Health-Analysis-UN-dataset-.git
cd Africa-Population-Trend-and-Health-Analysis-UN-dataset-
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Launch Jupyter Notebook**
```bash
jupyter notebook notebooks/main_analysis.ipynb
```

---

## 📊 How to Use

### Running the Analysis

1. Open `notebooks/main_analysis.ipynb` in Jupyter
2. Run cells sequentially from top to bottom
3. The notebook will:
   - Load and clean the data
   - Engineer features
   - Train three models for each hypothesis
   - Evaluate and compare model performance
   - Generate visualizations and metrics

### Using the Custom Modules

```python
import sys
sys.path.append('..')

from src.data_processor import DataProcessor
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator

# Load and clean data
processor = DataProcessor('df_africa_cleaned.xls')
df_clean = processor.process()

# Engineer features
fe = FeatureEngineer(df_clean)
df_features = fe.process()

# Train models
trainer = ModelTrainer(df_features, target='target_column', problem_type='regression')
trainer.train_test_split()
trainer.scale_numeric()
best_model, params = trainer.train_models()

# Evaluate models
evaluator = ModelEvaluator()
comparison = evaluator.evaluate_models(trainer, ['LinearRegression', 'RandomForestRegressor'], cv=5)
```

---

## 🔍 Methodology

### Data Processing
- **Source:** UN DESA population and health indicators
- **Coverage:** 54 African countries, 2010-2024
- **Missing Data:** Maternal mortality for 2024 preserved (systematic non-reporting)
- **Outliers:** Retained to preserve genuine demographic variation

### Feature Engineering
Created 8 new features:
- **4 Trend Features:** Measure change from 2010 to 2024
- **4 Mean Features:** Country-level averages across time period

### Models Implemented
For each hypothesis, three models were trained:
1. **Linear Regression** (baseline)
2. **Random Forest Regressor** (ensemble, captures non-linearity)
3. **Gradient Boosting Regressor** (boosting, sequential learning)

### Evaluation Metrics
- **R² Score:** Proportion of variance explained
- **RMSE:** Root Mean Squared Error
- **MAE:** Mean Absolute Error
- **Cross-Validation:** 5-fold CV for robustness

---

## 📈 Results Summary

| Hypothesis | Best Model | R² Score | Decision |
|------------|-----------|----------|----------|
| H1: Fertility → Maternal Mortality | Random Forest | 0.8325 | **ACCEPT** |
| H2: Life Expectancy → Pop. Growth | Random Forest | 0.7696 | **ACCEPT** |
| H3: Under-5 Mortality → Life Expect. | Random Forest | 0.9970 | **ACCEPT** |

**Key Insight:** Healthcare quality indicators (life expectancy, child mortality) are stronger predictors than fertility rates alone.

For comprehensive analysis, see [`HYPOTHESIS_TESTING_SUMMARY.md`](HYPOTHESIS_TESTING_SUMMARY.md)

---

## 📚 Dataset Description

**File:** `df_africa_cleaned.xls`

**Columns:**
- `code` - Country/region code
- `region_country_area` - Name of country or region
- `year` - Year of observation (2010-2024)
- `life_expectancy_at_birth_for_both_sexes_years` - Life expectancy (years)
- `life_expectancy_at_birth_for_females_years` - Female life expectancy (years)
- `life_expectancy_at_birth_for_males_years` - Male life expectancy (years)
- `maternal_mortality_ratio_deaths_per_100000_population` - Maternal deaths per 100,000
- `population_annual_rate_of_increase_percent` - Annual population growth (%)
- `total_fertility_rate_children_per_women` - Average children per woman
- `under_five_mortality_rate_for_both_sexes_per_1000_live_births` - Child deaths per 1,000

---

## 👥 Team & Contact

This is a group assignment for Practical Data Analytics course.

**Team Members:** Blaise Niyonkuru, Ajak Makuach Abuol, Akebert Tesfahu Arefaine, Ghea Sandrine Mawen

**Date Completed:** November 23, 2025

---

## 📄 License

This project is for academic purposes as part of the Practical Data Analytics course.

---

## 🙏 Acknowledgments

- UN DESA for providing population and health indicators
- Course instructors for assignment guidance
- Team members for collaborative effort