import unittest

import pandas as pd
from ml_helpers import get_year_quarter_combos


class TestGetYearQuarterCombos(unittest.TestCase):
    """Class testing the get_year_quarter_combos function in ml_helpers.
    It simply tests that given some inputs, the outputs are as expected."""

    def test_get_year_quarter_combos(self):
        start_year = 2020
        start_quarter = 2
        end_year = 2022
        end_quarter = 3

        expected_result = pd.DataFrame(
            {
                "Year": [2020, 2020, 2020, 2021, 2021, 2021, 2021, 2022, 2022, 2022],
                "Quarter": [2, 3, 4, 1, 2, 3, 4, 1, 2, 3],
            }
        )

        result = get_year_quarter_combos(start_year, start_quarter, end_year, end_quarter)

        self.assertTrue(expected_result.equals(result), "DataFrame generated is not as expected.")


if __name__ == "__main__":
    unittest.main()
