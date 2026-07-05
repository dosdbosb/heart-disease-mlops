"""
Unit tests for the data cleaning logic.
Uses small, hand-made sample data (not the real dataset) so tests
run instantly and don't depend on external files.
"""
import pandas as pd
import sys
import os

# Allows this test file to import from scripts/
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
from clean_data import clean_data  # noqa: E402


def make_sample_df():
    """A tiny fake dataset with the same columns as the real one."""
    return pd.DataFrame({
        "age": [63, 67, 41, 55],
        "sex": [1, 1, 0, 1],
        "cp": [1, 4, 2, 3],
        "trestbps": [145, 160, 130, 140],
        "chol": [233, 286, 204, 250],
        "fbs": [1, 0, 0, 0],
        "restecg": [2, 0, 1, 2],
        "thalach": [150, 108, 172, 160],
        "exang": [0, 1, 0, 0],
        "oldpeak": [2.3, 1.5, 1.4, 0.5],
        "slope": [3, 2, 1, 2],
        "ca": [0.0, 3.0, None, 1.0],   # one missing value on purpose
        "thal": [6.0, 3.0, 3.0, None],  # another missing value on purpose
        "num": [0, 2, 0, 1],           # includes a severity value > 1
    })


def test_target_is_binary():
    """After cleaning, 'num' should only ever contain 0 or 1, never 2/3/4."""
    df = clean_data(make_sample_df())
    assert set(df["num"].unique()).issubset({0, 1})


def test_missing_values_dropped():
    """Rows with missing 'ca' or 'thal' should be removed entirely."""
    df = clean_data(make_sample_df())
    assert df.isnull().sum().sum() == 0


def test_row_count_after_dropping():
    """Our sample has 2 rows with missing data out of 4, so 2 should remain."""
    df = clean_data(make_sample_df())
    assert len(df) == 2


def test_categorical_columns_encoded():
    """Original categorical columns should be gone, replaced by one-hot columns."""
    df = clean_data(make_sample_df())
    assert "cp" not in df.columns
    assert any(col.startswith("cp_") for col in df.columns)
