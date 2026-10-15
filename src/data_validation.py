from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/creditcard.csv")
EXPECTED_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the CSV, with a clear error if the file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download creditcard.csv from Kaggle "
            "and place it in data/raw/ (see data/README.md)."
        )
    return pd.read_csv(path)


def validate(df: pd.DataFrame) -> dict:
    """Run basic checks and return the results as a dictionary."""
    results = {}
    results["rows"] = len(df)
    results["columns_match"] = list(df.columns) == EXPECTED_COLUMNS
    results["missing_values"] = int(df.isna().sum().sum())
    results["duplicate_rows"] = int(df.duplicated().sum())
    results["negative_amounts"] = int((df["Amount"] < 0).sum())
    results["class_values"] = sorted(df["Class"].unique().tolist())
    results["fraud_count"] = int((df["Class"] == 1).sum())
    results["fraud_percent"] = round(100 * (df["Class"] == 1).mean(), 4)
    return results


if __name__ == "__main__":
    df = load_data()
    print("Data types:")
    print(df.dtypes.value_counts())
    print("\nValidation results:")
    for key, value in validate(df).items():
        print(f"  {key}: {value}")