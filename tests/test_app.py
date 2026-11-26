import unittest

from streamlit.testing.v1 import AppTest

from utils.data_loader import ROOT


class AppTests(unittest.TestCase):
    def test_navigation_prediction_reset_and_retrain(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 5)
        next(button for button in app.button if button.label == "Try a prediction").click().run()
        self.assertEqual(app.sidebar.radio[0].value, "Student Performance Prediction")
        self.assertFalse(app.exception)
        for page in app.sidebar.radio[0].options[1:]:
            with self.subTest(page=page):
                app.sidebar.radio[0].set_value(page).run()
                self.assertFalse(app.exception)
                self.assertIn(page, [heading.value for heading in app.main.title])
                if page == "Dataset Explorer":
                    app.text_input[0].set_value("1001").run()
                    self.assertEqual(len(app.dataframe[0].value), 1)
                    app.slider[0].set_value((95.0, 99.9)).run()
                    self.assertTrue(app.dataframe[0].value.empty)
                if page == "Model Training and Evaluation":
                    app.button[0].click().run()
                    self.assertFalse(app.exception)
                    self.assertIn("New model and evaluation files saved.", [item.value for item in app.success])
                if page == "Student Performance Prediction":
                    initial_gpa = app.number_input(key="input_previous_gpa").value
                    next(button for button in app.button if button.label == "Higher indicators").click().run()
                    self.assertGreater(app.number_input(key="input_previous_gpa").value, initial_gpa)
                    app.number_input(key="input_previous_gpa").set_value(9.9)
                    next(button for button in app.button if button.label == "Predict Performance").click().run()
                    self.assertFalse(app.exception)
                    self.assertTrue(any("Predicted performance" in item.value for item in app.markdown))
                    next(button for button in app.button if button.label == "Reset Inputs").click().run()
                    self.assertFalse(app.exception)
                    self.assertEqual(app.number_input(key="input_previous_gpa").value, initial_gpa)
                    self.assertFalse(any("Predicted performance" in item.value for item in app.markdown))

