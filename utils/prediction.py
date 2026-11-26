from math import isfinite

import pandas as pd

from utils.data_loader import FEATURES


def predict(pipeline, ranges, values):
    cleaned = {}
    for name, label in FEATURES.items():
        try:
            value = float(values[name])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Enter a number for {label}.") from error
        low, high = ranges[name]
        if not isfinite(value) or not low <= value <= high:
            raise ValueError(f"{label} must be between {low:g} and {high:g}.")
        if name == "backlogs" and not value.is_integer():
            raise ValueError("Number of backlogs must be a whole number.")
        cleaned[name] = value
    row = pd.DataFrame([cleaned], columns=list(FEATURES))
    category = pipeline.predict(row)[0]
    probabilities = {}
    if hasattr(pipeline, "predict_proba"):
        probabilities = dict(zip(pipeline.classes_, pipeline.predict_proba(row)[0]))
    return category, probabilities

