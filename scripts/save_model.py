"""
Trains the final Logistic Regression model on ALL the data (not just the
80% train split) and saves the model + scaler + feature list + threshold
as reusable files for the API to load later.
"""
import pandas as pd
import joblib
import json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import os

os.makedirs("models", exist_ok=True)

df = pd.read_csv("data/heart_disease_clean.csv")
X = df.drop(columns=["num"])
y = df["num"]

# Train on ALL available data for the final deployed model.
# (During Phase 2 we held back 20% to *evaluate* fairly — now that
# evaluation is done and we trust the model, we use every row we have
# to make the final version as well-trained as possible.)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

final_model = LogisticRegression(max_iter=1000, C=0.01)  # best_C found in Phase 2
final_model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(final_model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# Save the exact column order + the chosen threshold as a config file.
# The API will read this to know how to format incoming requests.
config = {
    "feature_columns": list(X.columns),
    "decision_threshold": 0.35
}
with open("models/model_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("Saved model.pkl, scaler.pkl, and model_config.json to models/")
print(f"\nFeature columns ({len(config['feature_columns'])}):")
print(config["feature_columns"])