import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset
import json
from pathlib import Path


def detect_drift(
    reference_path: str = "data/creditcard.csv",
    current_path: str = "data/recent_predictions.csv",
    drift_threshold: float = 0.3,
) -> dict:
    """
    Compare recent prediction inputs against training data.
    Returns drift report and whether drift was detected.

    Compatible with Evidently >= 0.7.x.
    """
    reference = pd.read_csv(reference_path).drop("Class", axis=1)
    current = pd.read_csv(current_path)

    # Use a sample of reference data
    reference_sample = reference.sample(n=min(1000, len(reference)), random_state=42)

    report = Report(metrics=[DataDriftPreset()])

    # In v0.7.x, run() returns a Snapshot object — capture it
    snapshot = report.run(reference_data=reference_sample, current_data=current)

    # Extract the dict from the Snapshot
    result = snapshot.dict()

    # Find the DriftedColumnsCount metric, which carries share + count
    drifted_columns_metric = next(
        (
            m for m in result["metrics"]
            if m.get("metric_name", "").startswith("DriftedColumnsCount")
        ),
        None,
    )

    if drifted_columns_metric is None:
        raise RuntimeError(
            "Could not find DriftedColumnsCount in report metrics. "
            f"Available metrics: {[m.get('metric_name') for m in result['metrics']]}"
        )

    drift_share: float = drifted_columns_metric["value"].get("share", 0.0)
    drift_detected: bool = drift_share >= drift_threshold

    print(f"Drift share:    {drift_share:.2f}")
    print(f"Threshold:      {drift_threshold:.2f}")
    print(f"Drift detected: {drift_detected}")

    # Save the visual HTML report (also on the Snapshot in v0.7.x)
    snapshot.save_html("drift_report.html")

    return {
        "drift_detected": drift_detected,
        "drift_share": drift_share,
        "threshold": drift_threshold,
    }


if __name__ == "__main__":
    result = detect_drift()
    exit(1 if result["drift_detected"] else 0)