from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset/student_performance.csv"
FEATURES = {
    "previous_gpa": "Previous GPA",
    "attendance_percentage": "Attendance percentage",
    "study_hours_per_week": "Study hours per week",
    "assignment_score": "Assignment score",
    "internal_marks": "Internal marks",
    "backlogs": "Number of backlogs",
}
TARGET = "performance_category"
CLASSES = ["Low", "Medium", "High"]
COLUMNS = ["student_id", *FEATURES, "final_percentage", TARGET]


def load_data(path=DATASET):
    frame = pd.read_csv(path)
    missing_columns = set(COLUMNS) - set(frame)
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}.")
    frame = frame[COLUMNS].copy()
    for column in COLUMNS[:-1]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if np.isinf(frame[column]).any():
            raise ValueError(f"{column} has infinite values.")
    if frame[TARGET].isna().any() or set(frame[TARGET]) != set(CLASSES):
        raise ValueError("The target must contain Low, Medium and High with no missing labels.")
    if frame.student_id.isna().any() or (frame.student_id % 1 != 0).any():
        raise ValueError("Student IDs must be present and integer-valued.")
    if (frame.backlogs.dropna() % 1 != 0).any():
        raise ValueError("Backlogs must be integer-valued.")
    if frame[list(FEATURES)].notna().sum().eq(0).any():
        raise ValueError("Every predictor needs at least one observed value.")
    return frame


def audit(frame):
    thresholds = pd.cut(frame.final_percentage, [-np.inf, 50, 75, np.inf],
                        right=False, labels=CLASSES)
    return {
        "missing": frame.isna().sum().to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_ids": int(frame.student_id.duplicated().sum()),
        "threshold_disagreements": int(
            (thresholds.notna() & thresholds.ne(frame[TARGET])).sum()
        ),
    }


def feature_ranges(frame):
    return {name: [float(frame[name].min()), float(frame[name].max())] for name in FEATURES}

