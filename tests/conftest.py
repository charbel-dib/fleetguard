import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification


@pytest.fixture
def classification_frame():
    x, y = make_classification(
        n_samples=240, n_features=8, n_informative=5, weights=[0.9, 0.1], random_state=42
    )
    frame = pd.DataFrame(x, columns=[f"sensor_{i}" for i in range(8)])
    frame.iloc[::9, 0] = np.nan
    frame["empty"] = np.nan
    return frame, pd.Series(y)
