from pathlib import Path
import pandas as pd

from data_validation import load_data

PROCESSED_DIR = Path("data/processed")
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15  # the remaining 15% is the test set

FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows and order transactions by time."""
    df = df.drop_duplicates()
    return df.sort_values("Time", kind="stable").reset_index(drop=True)


def split_chronological(df: pd.DataFrame, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    """Split an already time-ordered dataframe into train / validation / test."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def summarise(name: str, part: pd.DataFrame) -> None:
    fraud = int((part[TARGET_COLUMN] == 1).sum())
    print(
        f"{name:<10} rows={len(part):>7}  fraud={fraud:>4}  "
        f"fraud%={100 * fraud / len(part):.3f}  "
        f"time={part['Time'].min():.0f}-{part['Time'].max():.0f}"
    )


if __name__ == "__main__":
    raw = load_data()
    cleaned = clean(raw)
    print(f"Rows before: {len(raw)}  after removing duplicates: {len(cleaned)}\n")

    train, val, test = split_chronological(cleaned)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)
        summarise(name, part)

    print(f"\nSaved to {PROCESSED_DIR}")