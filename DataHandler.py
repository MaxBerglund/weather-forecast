import pandas as pd
import sys
import os
import pickle

# ---------- User settings ----------
PARQUET_PATH = "/home/maxb/rebase/hack/data/train.parquet"
# -----------------------------------

# Reads and displays basic information about a Parquet file.
def review_parquet(file_path, rows=5):
    print(f"Reviewing Parquet file: {file_path}")

    try:
        # Validate file existence
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read the Parquet file
        df = pd.read_parquet(file_path)
        
        # Display basic info
        print("\n--- File Preview ---")
        print(df.head(rows))  # First few rows
        
        print("\n--- Data Types ---")
        print(df.dtypes)
        
        print("\n--- Shape ---")
        print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
        print("\n--- Missing Values ---")
        print(df.isnull().sum())
        
    except FileNotFoundError as fnf_err:
        print(f"Error: {fnf_err}")
    except ValueError as val_err:
        print(f"Error reading Parquet file: {val_err}")
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)


def main():
    #review_parquet(PARQUET_PATH, rows=100)

    with open("/home/maxb/rebase/hack/model.pkl", "rb") as file:
        data = pickle.load(file)

    print(data)

if __name__ == "__main__":
    main()