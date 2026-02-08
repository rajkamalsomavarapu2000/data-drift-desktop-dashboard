import pandas as pd
from app.core.schema import infer_schema

def test_infer_schema_basic():
    df = pd.DataFrame({"a":[1,2,3], "b":["x","y","z"]})
    schema = infer_schema(df)
    kinds = {s.name: s.kind for s in schema}
    assert kinds["a"] == "numeric"
    assert kinds["b"] in ("categorical", "unknown")
