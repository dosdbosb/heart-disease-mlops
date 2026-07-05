"""
Generates EDA visualizations for the cleaned heart disease dataset:
1. Histograms of key numeric features
2. Correlation heatmap
3. Class balance plot
Saves all charts as PNGs in reports/figures/
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("reports/figures", exist_ok=True)

df = pd.read_csv("data/heart_disease_clean.csv")

# 1. Histograms of the main numeric (non-encoded) features
numeric_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
df[numeric_cols].hist(figsize=(12, 8), bins=20, edgecolor="black")
plt.suptitle("Distribution of Key Numeric Features")
plt.tight_layout()
plt.savefig("reports/figures/histograms.png")
plt.close()

# 2. Correlation heatmap (numeric features only, target included)
plt.figure(figsize=(10, 8))
corr = df[numeric_cols + ["num"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("reports/figures/correlation_heatmap.png")
plt.close()

# 3. Class balance
plt.figure(figsize=(6, 4))
sns.countplot(x="num", data=df)
plt.title("Class Balance: Disease (1) vs No Disease (0)")
plt.xlabel("Diagnosis")
plt.ylabel("Number of Patients")
plt.savefig("reports/figures/class_balance.png")
plt.close()

# 4. Feature relationship: how age and cholesterol relate to disease presence
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.boxplot(x="num", y="age", data=df, ax=axes[0])
axes[0].set_title("Age vs Disease Presence")
axes[0].set_xlabel("Diagnosis (0 = No Disease, 1 = Disease)")

sns.boxplot(x="num", y="chol", data=df, ax=axes[1])
axes[1].set_title("Cholesterol vs Disease Presence")
axes[1].set_xlabel("Diagnosis (0 = No Disease, 1 = Disease)")

plt.tight_layout()
plt.savefig("reports/figures/feature_relationships.png")
plt.close()


print("Saved 4 charts to reports/figures/")
print("\nClass balance counts:")
print(df["num"].value_counts())
