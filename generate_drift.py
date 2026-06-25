# generate_drift.py — creates fake "recent predictions" with shifted distribution
import pandas as pd
import numpy as np

df = pd.read_csv("data/creditcard.csv").drop("Class", axis=1)
sample = df.sample(500)

# simulate drift by shifting the distribution
sample["V1"] = sample["V1"] + np.random.normal(2, 1, len(sample))
sample["V2"] = sample["V2"] * 1.5
sample["Amount"] = sample["Amount"] * 3

sample.to_csv("data/recent_predictions.csv", index=False)
print("Drift data generated")