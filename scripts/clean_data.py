"""
Cleans the raw heart disease dataset:
1. Converts target to binary (0 = no disease, 1 = disease)
2. Drops rows with missing values
3. One-hot encodes categorical columns
Saves result to data/heart_disease_clean.csv
"""
import pandas as pd

df = pd.read_csv("data/heart_disease_raw.csv")

print("Missing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print(f"\nTotal rows before cleaning: {len(df)}")

# 1. Binary target: 0 stays 0, anything 1-4 becomes 1
df["num"] = (df["num"] > 0).astype(int)

# 2. Drop rows with any missing values
df = df.dropna()
print(f"Total rows after dropping missing values: {len(df)}")

# 3. One-hot encode true categorical columns
# (sex, fbs, exang are already binary 0/1, so we leave them alone)
categorical_cols = ["cp", "restecg", "slope", "thal", "ca"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

df.to_csv("data/heart_disease_clean.csv", index=False)
print(f"\nSaved cleaned data: {df.shape[0]} rows, {df.shape[1]} columns")
print(df.head())