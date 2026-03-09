import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_df():
    """Create a simple DataFrame for testing."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=n),
            "close": np.cumsum(np.random.randn(n)) + 100,
            "volume": np.abs(np.random.randn(n)) * 1000,
            "return_1d": np.random.randn(n) * 0.02,
        }
    )
