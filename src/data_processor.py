"""
data_processor.py

Minimal preprocessing pipeline for df_africa_cleaned.csv.

This class performs only:
- File loading
- Numeric type conversion
- String cleanup for region names
- Consistent lowercasing of column names

Strictly NO:
- Missing value filling
- Outlier handling
- Row dropping
- Transformations that alter the original data values

"""

import pandas as pd


class DataProcessor:
    """
    Load and minimally process the cleaned Africa dataset.

    This class ensures:
    - Consistent numeric types for indicator columns
    - Clean formatting for region names
    - Standardized column casing (lowercase)

    Parameters
    ----------
    file_path : str
        Path to the df_africa_cleaned.csv file.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    # ------------------------------------------------------------------
    def process(self) -> pd.DataFrame:
        """
        Load and minimally process the dataset without altering its values.

        Steps performed:
        - Load CSV
        - Convert numeric-like columns to numeric (coercing errors to NaN)
        - Strip whitespace from region names
        - Lowercase and clean column names

        Returns
        -------
        pandas.DataFrame
            The minimally processed dataset.

        Notes
        -----
        This method intentionally does *not*:
        - Fill missing values
        - Remove rows
        - Perform outlier filtering
        - Apply scaling or transformation
        """
        # Load the dataset
        try:
            df = pd.read_csv(self.file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {self.file_path}")

        # Convert indicator columns to numeric (except identifiers)
        for col in df.columns:
            if col not in ["code", "region_country_area", "year"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convert known numeric identifier fields
        df["code"] = pd.to_numeric(df["code"], errors="coerce")

        # 'year' must be numeric, integer type for modeling + grouping
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["year"] = df["year"].astype("Int64")  # preserves missing values safely

        # Clean region names (ensures formatting consistency)
        df["region_country_area"] = (
            df["region_country_area"]
            .astype(str)
            .str.strip()
        )

        # Standardize column names
        df.columns = [c.lower().strip() for c in df.columns]

        return df
