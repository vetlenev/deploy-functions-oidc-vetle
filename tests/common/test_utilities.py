import pytest

from common.utilities import thermal_resistance


@pytest.mark.unit
def test_thermal_resistance():
    import pandas as pd

    df = pd.DataFrame(
        {
            "T_hot_IN": [10, 11, 12],
            "T_hot_OUT": [3, 4, 6],
            "T_cold_IN": [1, 2, 3],
            "T_cold_OUT": [4, 4, 6],
            "Flow_hot": [2, 2, 4],
        }
    )
    result = df.apply(lambda x: thermal_resistance(x), axis=1)
    assert result == [1, 3, 4]  # CALCULATE AND CHANGE
