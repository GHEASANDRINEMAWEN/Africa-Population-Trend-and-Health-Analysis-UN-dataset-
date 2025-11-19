import pandas as pd

class DataProcessor:
    # Works on df_africa_cleaned.csv directly.
    # No filling, no outlier removal, no transformation.
    # Only ensures correct types and consistent formatting.

    def __init__(self, file_path: str):
        self.file_path = file_path

    def process(self):
       
        # Load the cleaned Africa dataset
      
        df = pd.read_csv(self.file_path)

        # Convert all numeric columns properly
      
        for col in df.columns:
            if col not in ["code", "region_country_area", "year"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["code"] = pd.to_numeric(df["code"], errors="coerce")
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)

       
        # Clean region names
       
        df["region_country_area"] = df["region_country_area"].astype(str).str.strip()

     
        # DO NOT fill maternal mortality 2024 missing values
        # DO NOT drop rows
        # DO NOT transform values
     

  
        # Ensure consistent column format
     
        df.columns = [c.lower().strip() for c in df.columns]

        return df
