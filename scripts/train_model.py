"""
Trains and evaluates two classifiers (Logistic Regression, Random Forest)
on the cleaned heart disease data, with MLflow tracking every run.
"""
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

mlflow.set_experiment("heart-disease-classification")

df = pd.read_csv("data/heart_disease_clean.csv")
X = df.drop(columns=["num"])
y = df["num"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def log_confusion_matrix(y_true, y_pred, filename):
    """Saves a confusion matrix plot and returns the path so we can log it."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(filename)
    plt.close()
    return filename


def log_roc_curve(y_true, y_proba, filename):
    """Saves an ROC curve plot and returns the path so we can log it."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(filename)
    plt.close()
    return filename


# --- Model 1: Logistic Regression ---
with mlflow.start_run(run_name="logistic_regression"):
    params = {"C": [0.01, 0.1, 1, 10]}
    grid = GridSearchCV(LogisticRegression(max_iter=1000), params, cv=5, scoring="roc_auc")
    grid.fit(X_train_scaled, y_train)
    model = grid.best_estimator_

    # Use our chosen 0.35 threshold instead of the default 0.5
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    threshold = 0.35
    y_pred = (y_proba >= threshold).astype(int)

    # Log parameters (the "settings" used)
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("best_C", grid.best_params_["C"])
    mlflow.log_param("decision_threshold", threshold)

    # Log metrics (the "scores")
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_proba))

    # Log artifacts (files: plots + the model itself)
    mlflow.log_artifact(log_confusion_matrix(y_test, y_pred, "reports/figures/cm_logreg.png"))
    mlflow.log_artifact(log_roc_curve(y_test, y_proba, "reports/figures/roc_logreg.png"))
    mlflow.sklearn.log_model(model, "model")

    print("Logistic Regression run logged.")

# --- Model 2: Random Forest ---
with mlflow.start_run(run_name="random_forest"):
    params = {"n_estimators": [100, 200], "max_depth": [None, 5, 10]}
    grid = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=5, scoring="roc_auc")
    grid.fit(X_train_scaled, y_train)
    model = grid.best_estimator_

    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)  # default 0.5 threshold for comparison model

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("best_n_estimators", grid.best_params_["n_estimators"])
    mlflow.log_param("best_max_depth", grid.best_params_["max_depth"])

    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_proba))

    mlflow.log_artifact(log_confusion_matrix(y_test, y_pred, "reports/figures/cm_rf.png"))
    mlflow.log_artifact(log_roc_curve(y_test, y_proba, "reports/figures/roc_rf.png"))
    mlflow.sklearn.log_model(model, "model")

    print("Random Forest run logged.")

print("\nAll runs logged. View them with: mlflow ui")