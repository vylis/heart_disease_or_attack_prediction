import unittest
import pandas as pd
import numpy as np
from utils.cleaning import clean_data


class test_clean_data(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                "A": [1, 2, np.nan, 4, 5, 5],
                "B": [np.nan, 2, 3, 4, 5, 5],
                "C": [1, 2, 3, 4, np.nan, np.nan],
            }
        )

    def test_clean_data(self):
        cleaned_df = clean_data(self.df)
        self.assertFalse(cleaned_df.isnull().values.any())
        self.assertFalse(cleaned_df.duplicated().any())
