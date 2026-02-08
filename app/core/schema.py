from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass(frozen=True)
class ColumnSchema:
    name: str
    kind: str  # "numeric" | "categorical" | "datetime" | "unknown"
    nunique: int
    missing_rate: float

def _try_parse_datetime(s: pd.Series) -> bool:
    if s.dtype.kind in ("M",):
        return True
    if s.dtype == object:
        sample = s.dropna().astype(str).head(50)
        if sample.empty:
            return False
        parsed = pd.to_datetime(sample, errors="coerce", utc=True)
        return parsed.notna().mean() >= 0.8
    return False

def infer_schema(df: pd.DataFrame, max_cat_unique: int = 50) -> list[ColumnSchema]:
    out: list[ColumnSchema] = []
    for col in df.columns:
        s = df[col]
        missing_rate = float(s.isna().mean())
        nunique = int(s.nunique(dropna=True))

        kind = "unknown"
        if _try_parse_datetime(s):
            kind = "datetime"
        else:
            if pd.api.types.is_numeric_dtype(s):
                kind = "numeric"
            elif pd.api.types.is_bool_dtype(s):
                kind = "categorical"
            else:
                # object-like: decide categorical vs unknown by cardinality
                kind = "categorical" if nunique <= max_cat_unique else "unknown"

        out.append(ColumnSchema(name=str(col), kind=kind, nunique=nunique, missing_rate=missing_rate))
    return out
