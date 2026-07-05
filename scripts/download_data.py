"""
Downloads the Heart Disease dataset from the UCI ML Repository (ID: 45)
and saves it as a clean CSV in the data/ folder.
"""
from ucimlrepo import fetch_ucirepo
import pandas as pd

# Fetch dataset directly from UCI's servers (no manual download needed)
heart_disease = fetch_ucirepo(id=45)

X = heart_disease.data.features   # the 13 input features (age, sex, cholesterol, etc.)
y = heart_disease.data.targets     # the target: 0 = no disease, 1-4 = disease severity

# Combine into a single dataframe and save
df = pd.concat([X, y], axis=1)
df.to_csv("data/heart_disease_raw.csv", index=False)

print(f"Saved {df.shape[0]} rows and {df.shape[1]} columns to data/heart_disease_raw.csv")
print(df.head())
