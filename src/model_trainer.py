import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class ModelTrainer:
    """
    Handles:
    - Train-test split (with identifier alignment)
    - Numeric feature scaling
    - Regression/classification model training
    - Hyperparameter tuning
    - Mapping predictions back to original entities
    """

    def __init__(self, df, target, problem_type="regression"):
        self.df = df
        self.target = target
        self.problem_type = problem_type

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.train_indices = None
        self.test_indices = None

        self.identifiers = None
        self.id_train = None
        self.id_test = None

        self.scaler = None
        self.best_model = None
        self.best_params = None

    # ---------------------------------------------------------------------
    # TRAIN-TEST SPLIT (with identifier alignment + stratification)
    # ---------------------------------------------------------------------
    def train_test_split(self, test_size=0.2, random_state=42):

        # Drop rows where target is missing (only affects maternal mortality)
        df_clean = self.df.dropna(subset=[self.target]).reset_index(drop=True)

        # Classification → drop ANY remaining NaNs (predictors)
        if self.problem_type == "classification":
            df_clean = df_clean.dropna().reset_index(drop=True)

        # Store identifiers
        self.identifiers = df_clean[["region_country_area", "year", "code"]].copy()

        # Separate X and y
        X = df_clean.drop(columns=[self.target])
        y = df_clean[self.target]

        # Keep only numeric predictors
        X = X.select_dtypes(include=["float64", "int64"])

        # Stratify if classification
        strat = y if self.problem_type == "classification" else None

        # Perform split and keep indices
        self.X_train, self.X_test, self.y_train, self.y_test, idx_train, idx_test = train_test_split(
            X,
            y,
            df_clean.index,
            test_size=test_size,
            random_state=random_state,
            stratify=strat
        )

        # Save indices for mapping
        self.train_indices = idx_train
        self.test_indices = idx_test

        # Match identifiers with X splits
        self.id_train = self.identifiers.loc[idx_train]
        self.id_test = self.identifiers.loc[idx_test]

        return self.X_train, self.X_test, self.y_train, self.y_test

    # ---------------------------------------------------------------------
    # SCALING
    # ---------------------------------------------------------------------
    def scale_numeric(self):
        numeric_cols = self.X_train.columns

        self.scaler = StandardScaler()
        self.X_train[numeric_cols] = self.scaler.fit_transform(self.X_train[numeric_cols])
        self.X_test[numeric_cols] = self.scaler.transform(self.X_test[numeric_cols])

        return self.X_train, self.X_test

    # ---------------------------------------------------------------------
    # MODEL CONFIGURATION
    # ---------------------------------------------------------------------
    def get_model_configs(self):
        if self.problem_type == "regression":
            return {
                "LinearRegression": (LinearRegression(), {}),
                "RandomForestRegressor": (
                    RandomForestRegressor(),
                    {"n_estimators": [200, 400], "max_depth": [None, 10, 20]}
                )
            }

        else:  # classification
            return {
                "LogisticRegression": (
                    LogisticRegression(max_iter=500),
                    {"C": [0.1, 1, 10]}
                ),
                "RandomForestClassifier": (
                    RandomForestClassifier(),
                    {"n_estimators": [200, 400], "max_depth": [None, 10, 20]}
                )
            }

    # ---------------------------------------------------------------------
    # MODEL TRAINING + HYPERPARAMETER TUNING
    # ---------------------------------------------------------------------
    def train_models(self, cv=5):
        configs = self.get_model_configs()
        best_score = -float("inf")

        for name, (model, params) in configs.items():
            print(f"\nTraining: {name}")

            try:
                # Models without hyperparameters (simple fit)
                if len(params) == 0:
                    model.fit(self.X_train, self.y_train)
                    score = model.score(self.X_test, self.y_test)
                    print(f"{name} Score = {score}")

                    if score > best_score:
                        best_score = score
                        self.best_model = model
                        self.best_params = "default"

                else:
                    # Hyperparameter tuning
                    grid = GridSearchCV(
                        estimator=model,
                        param_grid=params,
                        cv=cv,
                        scoring="r2" if self.problem_type == "regression" else "f1",
                        n_jobs=-1
                    )
                    grid.fit(self.X_train, self.y_train)

                    print(f"Best Params: {grid.best_params_}")
                    print(f"Best Score: {grid.best_score_}")

                    if grid.best_score_ > best_score:
                        best_score = grid.best_score_
                        self.best_model = grid.best_estimator_
                        self.best_params = grid.best_params_

            except Exception as e:
                print(f"Model {name} FAILED. Reason: {e}")

        # Final check
        if self.best_model is None:
            raise ValueError("No model successfully trained. Check feature NaNs or class balance.")

        print("\nFinal Best Model:")
        print(self.best_model)
        print("Parameters:", self.best_params)

        return self.best_model, self.best_params

    # ---------------------------------------------------------------------
    # MAP PREDICTIONS BACK TO IDENTIFIERS
    # ---------------------------------------------------------------------
    def map_predictions(self):
        if self.best_model is None:
            raise ValueError("Train a model before mapping predictions.")

        # Predict
        preds = self.best_model.predict(self.X_test)

        # Merge with identifiers
        results = self.id_test.copy()
        results["actual"] = self.y_test.values
        results["predicted"] = preds

        return results
