"""
Cleans the raw heart disease dataset:
1. Converts target to binary (0 = no disease, 1 = disease)
2. Drops rows with missing values
3. One-hot encodes categorical columns
Saves result to data/heart_disease_clean.csv
"""
import pandas as pd


def clean_data(df):
    """Takes a raw dataframe and returns a cleaned one. Testable in isolation."""
    df = df.copy()

    # 1. Binary target
    df["num"] = (df["num"] > 0).astype(int)

    # 2. Drop rows with missing values
    df = df.dropna()

    # 3. One-hot encode categorical columns
    categorical_cols = ["cp", "restecg", "slope", "thal", "ca"]
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    return df


if __name__ == "__main__":
    df = pd.read_csv("data/heart_disease_raw.csv")
    print("Missing values per column:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print(f"\nTotal rows before cleaning: {len(df)}")

    df_clean = clean_data(df)
    print(f"Total rows after dropping missing values: {len(df_clean)}")

    df_clean.to_csv("data/heart_disease_clean.csv", index=False)
    print(f"\nSaved cleaned data: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
    print(df_clean.head())