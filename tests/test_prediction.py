import tempfile
import unittest
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from train_model import train
from utils.data_loader import CLASSES, FEATURES, ROOT, TARGET, load_data
from utils.prediction import predict


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = joblib.load(ROOT / "models/best_model.joblib")
        cls.pipeline = cls.bundle["pipeline"]
        cls.ranges = cls.bundle["metadata"]["feature_ranges"]

    def test_feature_selection_and_no_leakage(self):
        self.assertEqual(list(self.pipeline.feature_names_in_), list(FEATURES))
        self.assertTrue({"final_percentage", "student_id", TARGET}.isdisjoint(
            self.pipeline.feature_names_in_))
        frame = load_data()
        x_train, _, _, _ = train_test_split(
            frame[list(FEATURES)], frame[TARGET], test_size=.2, stratify=frame[TARGET], random_state=42)
        self.assertEqual(list(self.pipeline.named_steps["impute"].statistics_),
                         list(x_train.median()))

    def test_retraining_and_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            table = train(output=temp)
            bundle = joblib.load(Path(temp) / "models/best_model.joblib")
        self.assertEqual(table.model.iloc[0], bundle["metadata"]["best_model"])
        self.assertEqual(bundle["metadata"]["test_size"], 60)
        self.assertEqual(bundle["metadata"]["train_size"], 240)
        for name, scores in bundle["results"]["models"].items():
            matrix = scores["confusion_matrix"]
            self.assertEqual(sum(map(sum, matrix)), 60)
            self.assertAlmostEqual(sum(matrix[i][i] for i in range(3)) / 60, scores["accuracy"])
            self.assertEqual(len(bundle["results"]["cross_validation"][name]["f1_macro"]), 5)

    def test_profiles_and_boundaries(self):
        profiles = [
            {key: limits[0] for key, limits in self.ranges.items()},
            {key: limits[1] for key, limits in self.ranges.items()},
            load_data()[list(FEATURES)].median().to_dict(),
        ]
        profiles[0]["backlogs"] = self.ranges["backlogs"][1]
        profiles[1]["backlogs"] = self.ranges["backlogs"][0]
        for profile in profiles:
            with self.subTest(profile=profile):
                label, probabilities = predict(self.pipeline, self.ranges, profile)
                self.assertIn(label, CLASSES)
                self.assertEqual(set(probabilities), set(CLASSES))
                self.assertAlmostEqual(sum(probabilities.values()), 1)

    def test_rejects_invalid_input(self):
        profile = load_data()[list(FEATURES)].median().to_dict()
        for key, value in (("backlogs", .5), ("internal_marks", 41),
                           ("previous_gpa", "bad"), ("attendance_percentage", float("nan"))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                predict(self.pipeline, self.ranges, {**profile, key: value})

