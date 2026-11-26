import tempfile
import unittest
from pathlib import Path

from utils.data_loader import COLUMNS, DATASET, FEATURES, audit, load_data
from utils.preprocessing import make_models


class DatasetTests(unittest.TestCase):
    def test_source_shape_and_quality(self):
        frame = load_data()
        self.assertEqual(frame.shape, (300, 9))
        self.assertEqual(list(frame), COLUMNS)
        quality = audit(frame)
        self.assertEqual(quality["threshold_disagreements"], 28)
        self.assertEqual(sum(quality["missing"].values()), 0)
        self.assertEqual(quality["duplicate_rows"], 0)
        self.assertEqual(quality["duplicate_ids"], 0)

    def test_missing_column(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / DATASET.name
            load_data().drop(columns="internal_marks").to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "internal_marks"):
                load_data(path)

    def test_missing_numeric_values_are_imputed(self):
        frame = load_data()
        frame.loc[0, "internal_marks"] = float("nan")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / DATASET.name
            frame.to_csv(path, index=False)
            loaded = load_data(path)
        self.assertEqual(audit(loaded)["missing"]["internal_marks"], 1)
        pipeline = make_models()["Logistic Regression"]
        pipeline.fit(loaded[list(FEATURES)], loaded.performance_category)
        self.assertEqual(len(pipeline.predict(loaded.loc[[0], list(FEATURES)])), 1)

