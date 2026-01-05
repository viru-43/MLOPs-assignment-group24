import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from src.data.preprocess import get_preprocessor

# Load dataset
df = pd.read_csv("data/raw/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing + model
preprocessor = get_preprocessor()
model = LogisticRegression(max_iter=1000)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# MLflow tracking
mlflow.set_experiment("heart_disease_prediction")

with mlflow.start_run():
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("roc_auc", roc)

    # Save model
    joblib.dump(pipeline, "models/heart_model.joblib")
    mlflow.sklearn.log_model(pipeline, "model")

    print("Model trained and logged in MLflow")
