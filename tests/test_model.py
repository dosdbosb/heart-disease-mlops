"""
Unit tests for the saved model artifacts (model, scaler, config).
Checks the files load correctly and produce valid predictions —
this is what would break if the API tried to use a corrupted or
incompatible saved model.
"""
import joblib
import json
import pandas as pd
import os

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def test_model_and_scaler_load():
    """The saved model and scaler files should load without error."""
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    assert model is not None
    assert scaler is not None


def test_config_has_required_keys():
    """model_config.json must contain the feature list and threshold,
    since the API depends on both to make a correct prediction."""
    with open(os.path.join(MODEL_DIR, "model_config.json")) as f:
        config = json.load(f)
    assert "feature_columns" in config
    assert "decision_threshold" in config
    assert 0 < config["decision_threshold"] < 1


def test_model_predicts_valid_probability():
    """Feeding one real row of data through scaler + model should give
    back a probability between 0 and 1 — not a crash, not a nonsense value."""
    model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "model_config.json")) as f:
        config = json.load(f)

    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "data", "heart_disease_clean.csv"))
    sample = df.drop(columns=["num"]).iloc[[0]]  # first row, as a 1-row dataframe

    sample_scaled = scaler.transform(sample[config["feature_columns"]])
    probability = model.predict_proba(sample_scaled)[0, 1]

    assert 0.0 <= probability <= 1.0
