# model_evaluator.py

Provides tools for evaluating, comparing, and visualizing the performance
of multiple regression models using standard performance metrics, cross-validation,
and feature importance interpretation.

## Class: ModelEvaluator

Evaluate and compare multiple regression models.

This class supports:
- Training models using train/test splits from a ModelTrainer instance
- Computing common regression metrics (R², RMSE, MAE)
- Performing cross-validation
- Storing feature importances (if provided by the model)
- Plotting model comparison and feature importance

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `results` | list of dict | Stores metric results for each evaluated model |
| `feature_importances` | dict | Maps model names to pandas.Series objects of feature importances |
| `available_models` | dict | Registry of model names mapped to their constructors |

### Methods

#### `__init__()`
Initialize the ModelEvaluator with empty results and available models.

```python
evaluator = ModelEvaluator()
```

#### `build_model(model_name)`
Create a model instance given its name.

**Parameters:**
- `model_name` (str): Name of the model as defined in available_models

**Returns:**
- estimator: Instantiated model with default hyperparameters

**Raises:**
- ValueError: If model_name is not recognized

**Available Models:**
- LinearRegression
- RandomForestRegressor
- GradientBoostingRegressor

```python
model = evaluator.build_model("LinearRegression")
```

#### `evaluate_single_model(name, model, X_train, X_test, y_train, y_test, cv=5)`
Train and evaluate a single regression model.

**Parameters:**
- `name` (str): Display name for the model
- `model` (estimator): Model instance to evaluate
- `X_train, X_test` (pandas.DataFrame): Training and testing features
- `y_train, y_test` (array-like): Training and testing target values
- `cv` (int, default=5): Number of cross-validation folds

**Stores:**
- R² (train + test)
- RMSE
- MAE
- Cross-validation mean + std
- Feature importance (if available)

```python
evaluator.evaluate_single_model(
    "LinearRegression",
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    cv=5
)
```

#### `evaluate_models(trainer, model_names, cv=5)`
Evaluate a list of models using a ModelTrainer instance.

**Parameters:**
- `trainer` (ModelTrainer): A fitted ModelTrainer object containing X_train, X_test, etc.
- `model_names` (list of str): Names of models to evaluate
- `cv` (int, default=5): Cross-validation folds

**Returns:**
- pandas.DataFrame: Table of evaluation results for all models

```python
evaluator = ModelEvaluator()
comparison = evaluator.evaluate_models(
    trainer,
    ["LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor"],
    cv=5
)
print(comparison)
```

#### `plot_comparison(metric="Test R2")`
Visualize a comparison of model performance using a selected metric.

**Parameters:**
- `metric` (str, default="Test R2"): The metric to plot (must exist in results)

**Available Metrics:**
- Train R2
- Test R2
- RMSE
- MAE
- CV Mean R2
- CV Std

```python
evaluator.plot_comparison("Test R2")
```

#### `plot_feature_importance(model_name, top_n=5)`
Plot top-N most important features for a given model.

**Parameters:**
- `model_name` (str): The name of the evaluated model
- `top_n` (int, default=5): Number of top features to display

**Note:** Only works with models that have feature importance attributes (LinearRegression, RandomForestRegressor, GradientBoostingRegressor)

```python
evaluator.plot_feature_importance("GradientBoostingRegressor", top_n=10)
```

## Usage Example

```python
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator

# Train models using ModelTrainer
trainer = ModelTrainer(df_features, target="maternal_mortality_ratio_deaths_per_100000_population")
trainer.train_test_split()
trainer.scale_numeric()
best_model, params = trainer.train_models()

# Evaluate all models
evaluator = ModelEvaluator()
comparison = evaluator.evaluate_models(
    trainer,
    ["LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor"],
    cv=5
)

# Display comparison table
print(comparison)

# Visualize results
evaluator.plot_comparison("Test R2")
evaluator.plot_feature_importance("GradientBoostingRegressor", top_n=10)
```

## Output Interpretation

### Metrics Explained

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **R² Score** | 1 - (SS_res / SS_tot) | Proportion of variance explained (0-1). Higher is better. |
| **RMSE** | √(MSE) | Root mean squared error. Lower is better. Same units as target. |
| **MAE** | Mean(\|y - ŷ\|) | Mean absolute error. Lower is better. Same units as target. |
| **CV Mean R2** | Mean of R² across folds | Average cross-validation performance. |
| **CV Std** | Std of R² across folds | Cross-validation stability. Lower is better. |

### Feature Importance

- **LinearRegression**: Coefficients (raw regression weights)
- **RandomForestRegressor**: Mean decrease in impurity
- **GradientBoostingRegressor**: Mean decrease in impurity (weighted)

Higher absolute values indicate stronger predictive influence.

## Integration with ModelTrainer

The ModelEvaluator works seamlessly with ModelTrainer:

```python
trainer = ModelTrainer(df_features, target="life_expectancy_at_birth_for_both_sexes_years")
trainer.train_test_split()
trainer.scale_numeric()
best_model, params = trainer.train_models()

# ModelTrainer automatically stores X_train, X_test, y_train, y_test
evaluator = ModelEvaluator()
results = evaluator.evaluate_models(trainer, ["LinearRegression", "RandomForestRegressor"], cv=5)
```