from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass(frozen=True)
class LoadResult:
    df: pd.DataFrame
    path: str

def load_csv(path: str, max_rows: int | None = None) -> LoadResult:
    if not path:
        raise ValueError("CSV path is empty.")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Could not read CSV: {e}") from e

    if max_rows is not None and len(df) > max_rows:
        df = df.head(max_rows).copy()

    if df.shape[1] == 0:
        raise ValueError("CSV has zero columns.")
    return LoadResult(df=df, path=path)

def validate_schema(baseline: pd.DataFrame, current: pd.DataFrame) -> tuple[bool, str]:
    bcols = list(baseline.columns)
    ccols = list(current.columns)
    if bcols != ccols:
        bset, cset = set(bcols), set(ccols)
        missing_in_current = sorted(list(bset - cset))
        extra_in_current = sorted(list(cset - bset))
        msg = "Schema mismatch.\n"
        if missing_in_current:
            msg += f"- Missing in current: {missing_in_current}\n"
        if extra_in_current:
            msg += f"- Extra in current: {extra_in_current}\n"
        msg += "- Column order differs or names differ."
        return False, msg
    return True, "Schema OK."
