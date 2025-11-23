"""
model_evaluator.py

Automatically evaluates models and saves:
- Performance tables
- Prediction outputs
- Comparison plots
- Feature importance plots

All results stored under:
results/
    ├── figures/
    ├── tables/
    └── predictions/
"""

import pandas as pd
import numpy as np
import os
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class ModelEvaluator:
    def __init__(self, results_dir="results"):
        self.results = []
        self.feature_importances = {}

        # registry of model types
        self.available_models = {
            "LinearRegression": LinearRegression,
            "RandomForestRegressor": RandomForestRegressor,
            "GradientBoostingRegressor": GradientBoostingRegressor,
        }

        # Create results folder structure
        self.results_dir = results_dir
        os.makedirs(f"{results_dir}/figures", exist_ok=True)
        os.makedirs(f"{results_dir}/tables", exist_ok=True)
        os.makedirs(f"{results_dir}/predictions", exist_ok=True)

    # ----------------------------------------------------------
    def build_model(self, model_name):
        if model_name not in self.available_models:
            raise ValueError(f"Unknown model: {model_name}")

        return self.available_models[model_name]()

    # ----------------------------------------------------------
    def evaluate_single_model(self, name, model, trainer, cv=5):
        X_train, X_test = trainer.X_train, trainer.X_test
        y_train, y_test = trainer.y_train, trainer.y_test

        model.fit(X_train, y_train)

        # predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mae = mean_absolute_error(y_test, y_test_pred)
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        # Cross validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")

        # store results
        self.results.append({
            "Model": name,
            "Train R2": r2_train,
            "Test R2": r2_test,
            "RMSE": rmse,
            "MAE": mae,
            "CV Mean R2": cv_scores.mean(),
            "CV Std": cv_scores.std()
        })

        # Feature importance
        if hasattr(model, "coef_"):
            self.feature_importances[name] = pd.Series(
                model.coef_, index=X_train.columns
            ).sort_values(ascending=False)

        elif hasattr(model, "feature_importances_"):
            self.feature_importances[name] = pd.Series(
                model.feature_importances_, index=X_train.columns
            ).sort_values(ascending=False)

        else:
            self.feature_importances[name] = None

        return model, y_test_pred

    # ----------------------------------------------------------
    def evaluate_models(self, trainer, model_names, save_name="results", cv=5):
        """
        Runs all models and automatically saves:
        - Model comparison table: results/tables/{save_name}_comparison.csv
        - Predictions for each model
        - Feature importance plot (if available)
        - Model comparison plot
        """

        self.results = []  # reset for each run
        all_predictions = {}

        for name in model_names:
            model = self.build_model(name)
            trained_model, preds = self.evaluate_single_model(name, model, trainer, cv=cv)
            all_predictions[name] = preds

        # Save table
        df_results = pd.DataFrame(self.results)
        df_results.to_csv(f"{self.results_dir}/tables/{save_name}_comparison.csv", index=False)

        # Save predictions per model
        pred_df = trainer.id_test.copy()
        for model_name, preds in all_predictions.items():
            pred_df[f"{model_name}_pred"] = preds

        pred_df.to_csv(f"{self.results_dir}/predictions/{save_name}_predictions.csv", index=False)

        # Save plots
        self.plot_comparison(metric="Test R2", save_path=f"{self.results_dir}/figures/{save_name}_model_comparison.png")
        self.save_all_feature_importances(save_name)

        return df_results

    # ----------------------------------------------------------
    def plot_comparison(self, metric="Test R2", save_path=None):
        df = pd.DataFrame(self.results)

        plt.figure(figsize=(8, 5))
        sns.barplot(data=df, x="Model", y=metric)
        plt.title(f"Model Comparison – {metric}")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.close()

    # ----------------------------------------------------------
    def save_all_feature_importances(self, save_name):
        for model_name, importance in self.feature_importances.items():
            if importance is None:
                continue

            plt.figure(figsize=(8, 6))
            sns.barplot(x=importance.head(10).values, y=importance.head(10).index)
            plt.title(f"Top Features – {model_name}")
            plt.tight_layout()

            plt.savefig(
                f"{self.results_dir}/figures/{save_name}_{model_name}_feature_importance.png",
                dpi=300
            )
            plt.close()
