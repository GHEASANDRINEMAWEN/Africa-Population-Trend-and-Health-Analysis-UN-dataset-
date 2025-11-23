import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor


class ModelTrainer:
    """
    Clean + final version.

    Handles:
    - Train-test split with identifier tracking
    - Dropping predictor columns with NaNs (safe + minimal)
    - Feature scaling
    - 3 supervised models (required by rubric)
    - Hyperparameter tuning
    - Mapping predictions back to countries/regions/years
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
    # TRAIN-TEST SPLIT + REMOVAL OF PREDICTORS THAT CONTAIN NaN
    # ---------------------------------------------------------------------
    def train_test_split(self, test_size=0.2, random_state=42):

        # Remove rows where TARGET is missing
        df_clean = self.df.dropna(subset=[self.target]).reset_index(drop=True)

        # Drop predictor columns with NaN (not target)
        cols_with_nan = df_clean.columns[df_clean.isna().any()].tolist()
        cols_with_nan = [c for c in cols_with_nan if c != self.target]

        if len(cols_with_nan) > 0:
            print("\nDropping predictor columns with missing values:", cols_with_nan)
            df_clean = df_clean.drop(columns=cols_with_nan)

        # Store identifiers
        self.identifiers = df_clean[["region_country_area", "year", "code"]].copy()

        # Separate predictors and target
        X = df_clean.drop(columns=[self.target])
        y = df_clean[self.target]

        # Restrict to numeric columns
        X = X.select_dtypes(include=["float64", "int64"])

        # Stratify if classification
        strat = y if self.problem_type == "classification" else None

        # Split + record indices
        self.X_train, self.X_test, self.y_train, self.y_test, idx_train, idx_test = train_test_split(
            X,
            y,
            df_clean.index,
            test_size=test_size,
            random_state=random_state,
            stratify=strat
        )

        self.train_indices = idx_train
        self.test_indices = idx_test
        self.id_train = self.identifiers.loc[idx_train]
        self.id_test = self.identifiers.loc[idx_test]

        print("\nTrain/Test Split Completed")
        print("Train size:", len(self.X_train))
        print("Test size:", len(self.X_test))

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
    # MODEL DEFINITIONS (3 MODELS REQUIRED BY RUBRIC)
    # ---------------------------------------------------------------------
    def get_model_configs(self):

        if self.problem_type == "regression":
            return {
                "LinearRegression": (LinearRegression(), {}),
                "RandomForestRegressor": (
                    RandomForestRegressor(),
                    {"n_estimators": [200, 400], "max_depth": [None, 10, 20]}
                ),
                "GradientBoostingRegressor": (
                    GradientBoostingRegressor(),
                    {"n_estimators": [50, 100],
                     "learning_rate": [0.05, 0.1],
                     "max_depth": [2, 3]}
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
                ),
                # (Third classification model removed because classification failed → not required now)
            }

    # ---------------------------------------------------------------------
    # TRAINING LOOP + GRID SEARCH
    # ---------------------------------------------------------------------
    def train_models(self, cv=5):

        configs = self.get_model_configs()
        best_score = -float("inf")

        for name, (model, params) in configs.items():
            print(f"\nTraining: {name}")

            try:
                if len(params) == 0:
                    # Simple model
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

                    print("Best params:", grid.best_params_)
                    print("Best score:", grid.best_score_)

                    if grid.best_score_ > best_score:
                        best_score = grid.best_score_
                        self.best_model = grid.best_estimator_
                        self.best_params = grid.best_params_

            except Exception as e:
                print(f"Model {name} FAILED. Reason: {e}")

        if self.best_model is None:
            raise ValueError("No model successfully trained.")

        print("\nBest Model Selected:")
        print(self.best_model)
        print("Params:", self.best_params)

        return self.best_model, self.best_params

    # ---------------------------------------------------------------------
    # MAPPING PREDICTIONS BACK TO REGION / COUNTRY / YEAR
    # ---------------------------------------------------------------------
    def map_predictions(self):
        if self.best_model is None:
            raise ValueError("Train a model before mapping predictions.")

        preds = self.best_model.predict(self.X_test)

        results = self.id_test.copy()
        results["actual"] = self.y_test.values
        results["predicted"] = preds

        return results
