"""
model_evaluator.py

Provides tools for evaluating, comparing, and visualizing the performance
of multiple regression models using standard performance metrics, cross-validation,
and feature importance interpretation.

"""

import pandas as pd
import numpy as np
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
    """
    Evaluate and compare multiple regression models.

    This class supports:
    - Training models using train/test splits from a ModelTrainer instance
    - Computing common regression metrics (R², RMSE, MAE)
    - Performing cross-validation
    - Storing feature importances (if provided by the model)
    - Plotting model comparison and feature importance

    Attributes
    ----------
    results : list of dict
        Stores metric results for each evaluated model.
    feature_importances : dict
        Maps model names to pandas.Series objects of feature importances.
    available_models : dict
        Registry of model names mapped to their constructors.
    """

    def __init__(self):
        self.results = []
        self.feature_importances = {}

        # Registry of supported models
        self.available_models = {
            "LinearRegression": LinearRegression,
            "RandomForestRegressor": RandomForestRegressor,
            "GradientBoostingRegressor": GradientBoostingRegressor,
        }

    # ------------------------------------------------------------------
    def build_model(self, model_name):
        """
        Create a model instance given its name.

        Parameters
        ----------
        model_name : str
            Name of the model as defined in available_models.

        Returns
        -------
        estimator
            Instantiated model with default hyperparameters.

        Raises
        ------
        ValueError
            If model_name is not recognized.
        """
        if model_name not in self.available_models:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available models: {list(self.available_models.keys())}"
            )

        return self.available_models[model_name]()

    # ------------------------------------------------------------------
    def evaluate_single_model(
        self, name, model, X_train, X_test, y_train, y_test, cv=5
    ):
        """
        Train and evaluate a single regression model.

        Parameters
        ----------
        name : str
            Display name for the model.
        model : estimator
            Model instance to evaluate.
        X_train, X_test : pandas.DataFrame
            Training and testing features.
        y_train, y_test : array-like
            Training and testing target values.
        cv : int, default=5
            Number of cross-validation folds.

        Stores
        ------
        - R² (train + test)
        - RMSE
        - MAE
        - Cross-validation mean + std
        - Feature importance (if available)
        """
        model.fit(X_train, y_train)

        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mae = mean_absolute_error(y_test, y_test_pred)
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        # Cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="r2")

        self.results.append(
            {
                "Model": name,
                "Train R2": r2_train,
                "Test R2": r2_test,
                "RMSE": rmse,
                "MAE": mae,
                "CV Mean R2": cv_scores.mean(),
                "CV Std": cv_scores.std(),
            }
        )

        # Extract feature importance
        if hasattr(model, "coef_"):  # linear models
            self.feature_importances[name] = pd.Series(
                model.coef_, index=X_train.columns
            ).sort_values(ascending=False)

        elif hasattr(model, "feature_importances_"):  # tree-based models
            self.feature_importances[name] = pd.Series(
                model.feature_importances_, index=X_train.columns
            ).sort_values(ascending=False)

        else:
            # No importance available
            self.feature_importances[name] = None

    # ------------------------------------------------------------------
    def evaluate_models(self, trainer, model_names, cv=5):
        """
        Evaluate a list of models using a ModelTrainer instance.

        Parameters
        ----------
        trainer : ModelTrainer
            A fitted ModelTrainer object containing X_train, X_test, etc.
        model_names : list of str
            Names of models to evaluate.
        cv : int, default=5
            Cross-validation folds.

        Returns
        -------
        pandas.DataFrame
            Table of evaluation results for all models.
        """
        for name in model_names:
            model = self.build_model(name)
            self.evaluate_single_model(
                name,
                model,
                trainer.X_train,
                trainer.X_test,
                trainer.y_train,
                trainer.y_test,
                cv=cv,
            )

        return pd.DataFrame(self.results)

    # ------------------------------------------------------------------
    def plot_comparison(self, metric="Test R2"):
        """
        Visualize a comparison of model performance using a selected metric.

        Parameters
        ----------
        metric : str, default="Test R2"
            The metric to plot (must exist in results).
        """
        df = pd.DataFrame(self.results)

        if metric not in df.columns:
            raise ValueError(
                f"Metric '{metric}' not found. Available metrics: {df.columns.tolist()}"
            )

        plt.figure(figsize=(8, 5))
        sns.barplot(data=df, x="Model", y=metric)
        plt.title(f"Model Comparison – {metric}")
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    def plot_feature_importance(self, model_name, top_n=5):
        """
        Plot top-N most important features for a given model.

        Parameters
        ----------
        model_name : str
            The name of the evaluated model.
        top_n : int, default=5
            Number of top features to display.
        """
        importance = self.feature_importances.get(model_name)

        if importance is None:
            print(f"No feature importance available for {model_name}.")
            return

        plt.figure(figsize=(8, 6))
        sns.barplot(
            x=importance.head(top_n).values, y=importance.head(top_n).index
        )
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title(f"Top {top_n} Features – {model_name}")
        plt.tight_layout()
        plt.show()
