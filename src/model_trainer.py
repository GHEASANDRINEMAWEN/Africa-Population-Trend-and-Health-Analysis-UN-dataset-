"""
model_trainer.py

A utility module containing the ModelTrainer class,
which automates model preparation, training, tuning, and prediction mapping.

Features:
- Train/test split with index tracking
- Automatic cleaning of NaN values in predictors
- Feature scaling (StandardScaler)
- Multiple model options (regression or classification)
- Hyperparameter tuning using GridSearchCV
- Prediction-to-identifier joining for interpretation

"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor
)


class ModelTrainer:
    """
    A training pipeline for regression or classification tasks.

    This class handles:
    - Cleaning data and removing predictors with missing values
    - Train/test splitting with preservation of identifying columns
    - Feature scaling using StandardScaler
    - Training 3 required models (per rubric) with optional hyperparameter tuning
    - Selecting the best model using validation scoring
    - Mapping predictions back to region/country/year for interpretability

    Parameters
    ----------
    df : pandas.DataFrame
        The full dataset.
    target : str
        Name of the target column.
    problem_type : str, default="regression"
        Either "regression" or "classification".
    """

    def __init__(self, df, target, problem_type="regression"):
        self.df = df
        self.target = target
        self.problem_type = problem_type

        # dataset splits
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        # track original dataset indices
        self.train_indices = None
        self.test_indices = None

        # identifier columns for mapping predictions
        self.identifiers = None
        self.id_train = None
        self.id_test = None

        # trained model + metadata
        self.scaler = None
        self.best_model = None
        self.best_params = None

    # ------------------------------------------------------------------
    def train_test_split(self, test_size=0.2, random_state=42):
        """
        Split the data into training and testing sets after cleaning.

        Steps:
        - Remove rows where the target is missing
        - Drop predictor columns containing NaN values
        - Select numeric predictors only
        - Return X_train, X_test, y_train, y_test

        Parameters
        ----------
        test_size : float, default=0.2
            Proportion of dataset to assign to testing.
        random_state : int, default=42
            Ensures reproducibility.

        Returns
        -------
        tuple(pd.DataFrame, pd.DataFrame, pd.Series, pd.Series)
        """
        # Remove entries where target value is missing
        df_clean = self.df.dropna(subset=[self.target]).reset_index(drop=True)

        # Identify predictors with missing values (non-target only)
        cols_with_nan = df_clean.columns[df_clean.isna().any()].tolist()
        cols_with_nan = [col for col in cols_with_nan if col != self.target]

        if cols_with_nan:
            print("\nDropping predictor columns with missing values:", cols_with_nan)
            df_clean = df_clean.drop(columns=cols_with_nan)

        # Save identifiers to merge back later
        try:
            self.identifiers = df_clean.loc[:, ["region_country_area", "year", "code"]]
        except KeyError:
            raise KeyError(
                "Expected identifier columns: ['region_country_area', 'year', 'code'] "
                "not found in dataframe."
            )

        # Split predictors and target
        X = df_clean.drop(columns=[self.target])
        y = df_clean[self.target]

        # Restrict to numeric features (avoids model issues)
        X = X.select_dtypes(include=["float64", "int64"])

        # Stratify only for classification tasks
        stratify = y if self.problem_type == "classification" else None

        # Perform the splitting while tracking original row indices
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.train_indices,
            self.test_indices,
        ) = train_test_split(
            X,
            y,
            df_clean.index,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        # Map identifiers to split sets
        self.id_train = self.identifiers.loc[self.train_indices]
        self.id_test = self.identifiers.loc[self.test_indices]

        print("\nTrain/Test Split Completed")
        print("Train size:", len(self.X_train))
        print("Test size:", len(self.X_test))

        return self.X_train, self.X_test, self.y_train, self.y_test

    # ------------------------------------------------------------------
    def scale_numeric(self):
        """
        Apply StandardScaler normalization to numeric columns in the dataset.

        Returns
        -------
        tuple(pd.DataFrame, pd.DataFrame)
            Scaled training and testing feature matrices.
        """
        numeric_cols = self.X_train.columns

        self.scaler = StandardScaler()

        # Fit-transform train, transform test
        self.X_train[numeric_cols] = self.scaler.fit_transform(
            self.X_train[numeric_cols]
        )
        self.X_test[numeric_cols] = self.scaler.transform(self.X_test[numeric_cols])

        return self.X_train, self.X_test

    # ------------------------------------------------------------------
    def get_model_configs(self):
        """
        Retrieve model configurations and hyperparameter grids.

        Returns
        -------
        dict
            Keys are model names, values are (estimator, param_grid) pairs.
        """
        if self.problem_type == "regression":
            return {
                "LinearRegression": (LinearRegression(), {}),
                "RandomForestRegressor": (
                    RandomForestRegressor(),
                    {"n_estimators": [200, 400], "max_depth": [None, 10, 20]},
                ),
                "GradientBoostingRegressor": (
                    GradientBoostingRegressor(),
                    {
                        "n_estimators": [50, 100],
                        "learning_rate": [0.05, 0.1],
                        "max_depth": [2, 3],
                    },
                ),
            }

        return {
            "LogisticRegression": (
                LogisticRegression(max_iter=500),
                {"C": [0.1, 1, 10]},
            ),
            "RandomForestClassifier": (
                RandomForestClassifier(),
                {"n_estimators": [200, 400], "max_depth": [None, 10, 20]},
            ),
        }

    # ------------------------------------------------------------------
    def train_models(self, cv=5):
        """
        Train each model, run hyperparameter tuning, and choose the best one.

        Parameters
        ----------
        cv : int, default=5
            Number of cross-validation folds.

        Returns
        -------
        tuple(model, dict or str)
            Best model and best parameters.
        """
        configs = self.get_model_configs()
        best_score = -float("inf")

        for name, (model, params) in configs.items():
            print(f"\nTraining: {name}")

            try:
                # Simple model: no hyperparameter grid
                if not params:
                    model.fit(self.X_train, self.y_train)
                    score = model.score(self.X_test, self.y_test)

                    print(f"{name} Score = {score:.4f}")

                    if score > best_score:
                        best_score = score
                        self.best_model = model
                        self.best_params = "default"

                # Model with hyperparameter search
                else:
                    grid = GridSearchCV(
                        estimator=model,
                        param_grid=params,
                        cv=cv,
                        scoring="r2" if self.problem_type == "regression" else "f1",
                        n_jobs=-1,
                    )
                    grid.fit(self.X_train, self.y_train)

                    print("Best params:", grid.best_params_)
                    print("Best score:", grid.best_score_)

                    if grid.best_score_ > best_score:
                        best_score = grid.best_score_
                        self.best_model = grid.best_estimator_
                        self.best_params = grid.best_params_

            except Exception as error:
                print(f"Model {name} FAILED. Reason: {error}")

        if self.best_model is None:
            raise ValueError("No model successfully trained.")

        print("\nBest Model Selected:")
        print(self.best_model)
        print("Params:", self.best_params)

        return self.best_model, self.best_params

    # ------------------------------------------------------------------
    def map_predictions(self):
        """
        Generate predictions and combine them with identifiers.

        Returns
        -------
        pandas.DataFrame
            A results dataframe containing actual, predicted, and identifiers.

        Raises
        ------
        ValueError
            If called before a model is trained.
        """
        if self.best_model is None:
            raise ValueError("Train a model before calling map_predictions().")

        predictions = self.best_model.predict(self.X_test)

        results = self.id_test.copy()
        results["actual"] = self.y_test.values
        results["predicted"] = predictions

        return results
