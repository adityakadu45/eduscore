import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock

import joblib
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from utils.data_loader import CLASSES, DATASET, FEATURES, ROOT, TARGET, audit, feature_ranges, load_data
from utils.preprocessing import SEED, make_models

PUBLISH_LOCK = Lock()


def train(data_path=DATASET, output=ROOT):
    output = Path(output)
    frame = load_data(data_path)
    quality = audit(frame)
    clean = frame.drop_duplicates()
    if clean.student_id.duplicated().any():
        raise ValueError("Conflicting records share a student ID. Resolve these before training.")
    x_train, x_test, y_train, y_test = train_test_split(
        clean[list(FEATURES)], clean[TARGET], test_size=0.20,
        random_state=SEED, stratify=clean[TARGET],
    )
    if y_train.value_counts().reindex(CLASSES, fill_value=0).min() < 5:
        raise ValueError("Each class needs at least five training rows for stratified CV.")
    if set(y_test) != set(CLASSES):
        raise ValueError("Every class must appear in the holdout.")
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    candidates = make_models()
    cv = {}
    for name, pipeline in candidates.items():
        scores = cross_validate(pipeline, x_train, y_train, cv=folds,
                                scoring=["f1_macro", "accuracy"], error_score="raise")
        cv[name] = {metric: scores[f"test_{metric}"].tolist()
                    for metric in ("f1_macro", "accuracy")}
    cv_table = pd.DataFrame([
        {"model": name, "cv_macro_f1": sum(scores["f1_macro"]) / 5,
         "cv_accuracy": sum(scores["accuracy"]) / 5}
        for name, scores in cv.items()
    ]).sort_values(["cv_macro_f1", "cv_accuracy", "model"], ascending=[False, False, True])
    best = cv_table.iloc[0]["model"]
    evaluations, reports = {}, {}
    for name, pipeline in candidates.items():
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        report = classification_report(y_test, predictions, labels=CLASSES,
                                       zero_division=0, output_dict=True)
        reports[name] = report
        evaluations[name] = {
            "accuracy": accuracy_score(y_test, predictions),
            **{f"{prefix}_{key}": report[average][metric]
               for prefix, average in (("macro", "macro avg"), ("weighted", "weighted avg"))
               for key, metric in (("precision", "precision"), ("recall", "recall"), ("f1", "f1-score"))},
            "confusion_matrix": confusion_matrix(y_test, predictions, labels=CLASSES).tolist(),
        }
    importance = permutation_importance(candidates[best], x_test, y_test,
                                        scoring="f1_macro", n_repeats=20, random_state=SEED)
    baseline = y_train.value_counts().idxmax()
    metadata = {
        "best_model": best, "selection": "Mean training CV macro F1, then CV accuracy and model name",
        "features": list(FEATURES), "classes": CLASSES, "feature_ranges": feature_ranges(frame),
        "train_size": len(x_train), "test_size": len(x_test), "random_state": SEED,
        "train_classes": y_train.value_counts().to_dict(), "test_classes": y_test.value_counts().to_dict(),
        "majority_class": baseline, "baseline_accuracy": float(y_test.eq(baseline).mean()),
        "dataset_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "sklearn_version": sklearn.__version__, "trained_at": datetime.now(timezone.utc).isoformat(),
        "audit": quality,
    }
    result = {
        "models": evaluations, "cross_validation": cv,
        "importance": {key: {"mean": float(mean), "std": float(std)}
                       for key, mean, std in zip(FEATURES, importance.importances_mean,
                                                 importance.importances_std)},
    }
    numbers = pd.DataFrame.from_dict(evaluations, orient="index").drop(columns="confusion_matrix")
    comparison = cv_table.join(numbers, on="model")
    # Stage beside each destination so replaced files inherit its directory permissions.
    staged = {}
    for folder in ("results", "models"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    try:
        for relative in ("results/model_comparison.csv", "models/model_metadata.json",
                         "results/classification_reports.json", "results/evaluation_results.json",
                         "models/candidate_models.joblib", "models/best_model.joblib"):
            destination = output / relative
            with NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as temp:
                staged[relative] = Path(temp.name)
        comparison.to_csv(staged["results/model_comparison.csv"], index=False)
        for relative, contents in {
            "models/model_metadata.json": metadata,
            "results/classification_reports.json": reports,
            "results/evaluation_results.json": result,
        }.items():
            staged[relative].write_text(json.dumps(contents, indent=2), encoding="utf-8")
        joblib.dump(candidates, staged["models/candidate_models.joblib"], compress=3)
        joblib.dump({"pipeline": candidates[best], "metadata": metadata, "results": result,
                     "reports": reports, "comparison": comparison},
                    staged["models/best_model.joblib"], compress=3)
        # Publish the self-contained app bundle after the supporting files.
        with PUBLISH_LOCK:
            for relative, temporary in staged.items():
                temporary.replace(output / relative)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
    return comparison


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate and train EduScore models.")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        data = load_data()
        print(f"Loaded {len(data)} rows and {len(data.columns)} columns.")
        print(json.dumps(audit(data), indent=2))
    else:
        print(train().round(4).to_string(index=False))

