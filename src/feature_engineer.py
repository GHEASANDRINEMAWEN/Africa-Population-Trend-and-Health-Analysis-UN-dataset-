import pandas as pd
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    # Takes in the cleaned df and generates new features:
    # - Fertility category (high vs low)
    # - Trend features (change from 2010 to 2024)
    # - Averages
    # - Normalized numeric features (optional for modeling)

    def __init__(self, df):
        self.df = df.copy()


    def add_trend_features(self):
        # Compute change between 2010 and 2024 for each country
        trend_cols = [
            "life_expectancy_at_birth_for_both_sexes_years",
            "total_fertility_rate_children_per_women",
            "under_five_mortality_rate_for_both_sexes_per_1000_live_births",
            "population_annual_rate_of_increase_percent"
        ]

        for col in trend_cols:
            self.df[f"{col}_change_2010_2024"] = (
                self.df.groupby("region_country_area")[col].transform(
                    lambda x: x.iloc[-1] - x.iloc[0]
                )
            )

        return self.df

    def add_means(self):
        # Average indicators per country (2010–2024)
        mean_cols = [
            "life_expectancy_at_birth_for_both_sexes_years",
            "total_fertility_rate_children_per_women",
            "under_five_mortality_rate_for_both_sexes_per_1000_live_births",
            "population_annual_rate_of_increase_percent"
        ]

        for col in mean_cols:
            self.df[f"{col}_mean"] = self.df.groupby("region_country_area")[col].transform("mean")

        return self.df

    def normalize_numeric(self):
        # Standard scaling of numeric columns
        numeric_cols = self.df.select_dtypes(include=["float64", "int64"]).columns
        scaler = StandardScaler()
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])
        return self.df

    def process(self):
        # Run all feature creation steps
        self.add_trend_features()
        self.add_means()
        return self.df
