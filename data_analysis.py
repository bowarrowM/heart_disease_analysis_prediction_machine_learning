import pandas as pd

from all_imports import CSV_PATH


def run_eda(df: pd.DataFrame) -> None:
    print("=== Head ===\n", df.head(), "\n")
    print("=== Info ===")
    df.info()
    print("\n=== Describe ===\n", df.describe(), "\n")
    print("=== Columns ===\n", df.columns.tolist(), "\n")
    print("=== Missing Values ===\n", df.isnull().sum(), "\n")
    print("=== Duplicates ===\n", df.duplicated().sum(), "\n")


if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)
    run_eda(df)
