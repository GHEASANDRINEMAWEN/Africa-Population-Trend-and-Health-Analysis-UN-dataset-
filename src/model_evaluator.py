import pandas as pd
import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import all regression models here
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


class ModelEvaluator:
    """
    Evaluates and compares multiple regression models.
    """

    def __init__(self):
        self.results = []
        self.feature_importances = {}

        # Dictionary for easy model creation
        self.available_models = {
            "LinearRegression": LinearRegression,
            "RandomForestRegressor": RandomForestRegressor,
            "GradientBoostingRegressor": GradientBoostingRegressor
        }

    # -----------------------------------------------------------
    # Build model instances with default hyperparameters
    # -----------------------------------------------------------
    def build_model(self, model_name):
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not found in available models.")

        ModelClass = self.available_models[model_name]
        return ModelClass()

    # -----------------------------------------------------------
    # Evaluate one model
    # -----------------------------------------------------------
    def evaluate_single_model(self, name, model, X_train, X_test, y_train, y_test, cv=5):
        # Make copies to avoid issues with modified data
        X_train_clean = X_train.copy()
        X_test_clean = X_test.copy()
        
        # Drop any columns with NaN values
        cols_with_nan = X_train_clean.columns[X_train_clean.isna().any()].tolist()
        if cols_with_nan:
            print(f"  Dropping columns with NaN for {name}: {cols_with_nan}")
            X_train_clean = X_train_clean.drop(columns=cols_with_nan)
            X_test_clean = X_test_clean.drop(columns=cols_with_nan)
        
        model.fit(X_train_clean, y_train)

        y_train_pred = model.predict(X_train_clean)
        y_test_pred = model.predict(X_test_clean)

        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        mae = mean_absolute_error(y_test, y_test_pred)
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        cv_scores = cross_val_score(model, X_train_clean, y_train, cv=cv, scoring="r2")

        self.results.append({
            "Model": name,
            "Train R2": r2_train,
            "Test R2": r2_test,
            "RMSE": rmse,
            "MAE": mae,
            "CV Mean R2": cv_scores.mean(),
            "CV Std": cv_scores.std()
        })

        # Store importance
        if hasattr(model, "coef_"):
            self.feature_importances[name] = pd.Series(
                model.coef_, index=X_train_clean.columns
            ).sort_values(ascending=False)

        elif hasattr(model, "feature_importances_"):
            self.feature_importances[name] = pd.Series(
                model.feature_importances_, index=X_train_clean.columns
            ).sort_values(ascending=False)

        else:
            self.feature_importances[name] = None

    # -----------------------------------------------------------
    # Evaluate a list of models
    # -----------------------------------------------------------
    def evaluate_models(self, trainer, model_names, cv=5):
        for name in model_names:
            model = self.build_model(name)
            self.evaluate_single_model(
                name,
                model,
                trainer.X_train,
                trainer.X_test,
                trainer.y_train,
                trainer.y_test,
                cv=cv
            )

        return pd.DataFrame(self.results)


    # Visualization helpers
 
    def plot_comparison(self, metric="Test R2"):
        df = pd.DataFrame(self.results)

        plt.figure(figsize=(8, 5))
        sns.barplot(data=df, x="Model", y=metric)
        plt.title(f"Model Comparison – {metric}")
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self, model_name, top_n=5):
        importance = self.feature_importances.get(model_name)

        if importance is None:
            print(f"No feature importance available for {model_name}")
            return

        plt.figure(figsize=(8, 6))
        sns.barplot(
            x=importance.head(top_n).values,
            y=importance.head(top_n).index
        )
        plt.ylabel("Features")
        plt.title(f"Top {top_n} Features – {model_name}")
        plt.tight_layout()
        plt.show()
