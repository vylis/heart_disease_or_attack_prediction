import unittest
import pandas as pd
from utils.preprocessing import process_data


class test_process_data(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "HighBP": [0, 1],
                "HighChol": [0, 1],
                "CholCheck": [0, 1],
                "BMI": [21, 25],
                "Smoker": [1, 0],
                "Stroke": [0, 1],
                "Diabetes": [0, 2],
                "PhysActivity": [1, 0],
                "Fruits": [1, 0],
                "Veggies": [0, 1],
                "HvyAlcoholConsump": [0, 1],
                "AnyHealthcare": [1, 0],
                "NoDocbcCost": [0, 1],
                "GenHlth": [2, 1],
                "MentHlth": [4, 3],
                "PhysHlth": [2, 3],
                "DiffWalk": [0, 1],
                "Sex": [1, 0],
                "Age": [3, 2],
                "Education": [3, 2],
                "Income": [2, 1],
            }
        )

    def test_process_data(self):
        processed_df = process_data(self.df, scaling_method="min_max")
        self.assertIsInstance(processed_df, pd.DataFrame)
        self.assertEqual(processed_df.shape[1], self.df.shape[1])
