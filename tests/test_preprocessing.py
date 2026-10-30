import pandas as pd

from data_validation import EXPECTED_COLUMNS, validate
from preprocess import clean, split_chronological


def frame(times):
    return pd.DataFrame({"Time": times, "Amount": [1.0] * len(times), "Class": [0] * len(times)})


def test_clean_removes_exact_duplicates():
    df = pd.DataFrame({"Time": [1, 1, 2], "Amount": [5.0, 5.0, 7.0], "Class": [0, 0, 1]})
    assert len(clean(df)) == 2


def test_clean_orders_by_time():
    assert clean(frame([3, 1, 2]))["Time"].tolist() == [1, 2, 3]


def test_split_sizes_and_no_overlap():
    train, val, test = split_chronological(clean(frame(list(range(100)))))
    assert len(train) + len(val) + len(test) == 100
    assert abs(len(train) - 70) <= 1 and abs(len(val) - 15) <= 1
    assert set(train.index).isdisjoint(val.index) and set(val.index).isdisjoint(test.index)


def test_split_is_chronological():
    train, val, test = split_chronological(clean(frame(list(range(100)))))
    assert train["Time"].max() <= val["Time"].min()
    assert val["Time"].max() <= test["Time"].min()


def test_validate_counts_problems():
    row = [0.0] * len(EXPECTED_COLUMNS)
    df = pd.DataFrame([row, row, row], columns=EXPECTED_COLUMNS)
    df.loc[2, "Amount"] = -1.0
    result = validate(df)
    assert result["rows"] == 3
    assert result["columns_match"] is True
    assert result["duplicate_rows"] == 1
    assert result["negative_amounts"] == 1