"""
feature_engineer.py

Tools for generating derived features from demographic datasets.

The FeatureEngineer class performs:
- Trend computation (change from first to last year per country)
- Country-level averages
- Optional numeric normalization (StandardScaler)
- Flexible, safe feature transformations with error handling

"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """
    A class for creating additional analytic and model-ready features.

    Parameters
    ----------
    df : pandas.DataFrame
        The cleaned dataset containing yearly country-level indicators.

    Notes
    -----
    Required columns for trend and mean features:
        - "region_country_area"
        - life_expectancy_at_birth_for_both_sexes_years
        - total_fertility_rate_children_per_women
        - under_five_mortality_rate_for_both_sexes_per_1000_live_births
        - population_annual_rate_of_increase_percent
    """

    # Columns required for trends and means
    INDICATOR_COLS = [
        "life_expectancy_at_birth_for_both_sexes_years",
        "total_fertility_rate_children_per_women",
        "under_five_mortality_rate_for_both_sexes_per_1000_live_births",
        "population_annual_rate_of_increase_percent",
    ]

    def __init__(self, df):
        if "region_country_area" not in df.columns:
            raise KeyError("DataFrame must contain the 'region_country_area' column.")

        missing = [col for col in self.INDICATOR_COLS if col not in df.columns]
        if missing:
            raise KeyError(f"Missing required indicator columns: {missing}")

        # Work on a copy to avoid modifying original data
        self.df = df.copy()

    # ------------------------------------------------------------
    def add_trend_features(self):
        """
        Add trend features showing the change from first available
        year to last available year for each indicator.

        Returns
        -------
        pandas.DataFrame
            Updated dataframe with *_change_2010_2024 columns.

        Notes
        -----
        Uses groupby().transform() so the result aligns with the original shape.
        """
        for col in self.INDICATOR_COLS:
            new_col = f"{col}_change_2010_2024"

            # Compute change as (last value - first value) for each country group
            self.df[new_col] = self.df.groupby("region_country_area")[col].transform(
                lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0
            )

        return self.df

    # ------------------------------------------------------------
    def add_means(self):
        """
        Add country-level averages across all years for each indicator.

        Returns
        -------
        pandas.DataFrame
            Updated dataframe with *_mean columns.
        """
        for col in self.INDICATOR_COLS:
            self.df[f"{col}_mean"] = self.df.groupby("region_country_area")[col].transform(
                "mean"
            )
        return self.df

    # ------------------------------------------------------------
    def normalize_numeric(self):
        """
        Normalize numeric columns using StandardScaler.

        Returns
        -------
        pandas.DataFrame
            Updated dataframe with all numeric values standardized.

        Notes
        -----
        Scaling occurs across the entire dataset, not per-country.
        """
        numeric_cols = self.df.select_dtypes(include=["float64", "int64"]).columns

        if numeric_cols.empty:
            print("Warning: No numeric columns found to normalize.")
            return self.df

        scaler = StandardScaler()
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])

        return self.df

    # ------------------------------------------------------------
    def process(self):
        """
        Run all feature engineering steps except normalization.

        Returns
        -------
        pandas.DataFrame
            Updated dataframe containing all engineered features.
        """
        self.add_trend_features()
        self.add_means()
        return self.df
