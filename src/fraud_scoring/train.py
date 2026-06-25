import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
import joblib
import json
from pathlib import Path

BASELINE_PATH = "models/baseline_metrics.json"

def load_baseline() -> dict:
    if Path(BASELINE_PATH).exists():
        with open(BASELINE_PATH) as f:
            return json.load(f)
    return {"f1": 0.0}  # no baseline yet — anything beats this

def save_baseline(metrics: dict):
    Path("models").mkdir(exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(metrics, f)

def train(data_path: str = "data/creditcard.csv"):
    mlflow.set_experiment("fraud-scoring")

    df = pd.read_csv(data_path)
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run():
        params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
        mlflow.log_params(params)

        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        metrics = {
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        # baseline comparison
        baseline = load_baseline()

        print(f"F1:        {metrics['f1']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"Baseline F1: {baseline['f1']:.4f}")

        if metrics["f1"] > baseline["f1"]:
            print("✓ New model beats baseline — saving")
            Path("models").mkdir(exist_ok=True)
            joblib.dump(model, "models/model.joblib")
            save_baseline(metrics)
            return {"promoted": True, **metrics}
        else:
            print("✗ New model does NOT beat baseline — keeping existing model")
            return {"promoted": False, **metrics}

if __name__ == "__main__":
    result = train()
    # exit code 1 if not promoted — GitHub Actions reads this
    import sys
    sys.exit(0 if result["promoted"] else 1)