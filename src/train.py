import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

DATA_PATH = "data/cleaned/heart_clean.csv"

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y

def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_prob))
    }

def main():
    X, y = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    experiments = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000))
        ]),
        "RandomForest": Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=200,
                random_state=42
            ))
        ])
    }

    mlflow.set_experiment("Heart_Disease_Classification")

    for name, model in experiments.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)

            metrics = evaluate(model, X_train, X_test, y_train, y_test)
            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(model, artifact_path="model")

            print(f"\n{name} Results:")
            print(metrics)

if __name__ == "__main__":
    main()
