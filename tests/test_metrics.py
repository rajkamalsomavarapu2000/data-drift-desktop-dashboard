import pandas as pd
from app.core.drift_metrics import psi_numeric

def test_psi_zero_when_same():
    b = pd.Series([1,2,3,4,5,6,7,8,9,10])
    c = pd.Series([1,2,3,4,5,6,7,8,9,10])
    assert abs(psi_numeric(b, c)) < 1e-6
