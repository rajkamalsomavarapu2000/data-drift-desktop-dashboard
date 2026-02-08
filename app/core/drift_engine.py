from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
from app.core.schema import infer_schema, ColumnSchema
from app.core import drift_metrics as dm

@dataclass(frozen=True)
class FeatureDrift:
    name: str
    kind: str
    missing_delta: float
    score: float
    details: dict

def compute_drift(baseline: pd.DataFrame, current: pd.DataFrame) -> tuple[list[ColumnSchema], list[FeatureDrift]]:
    schema = infer_schema(baseline)
    drifts: list[FeatureDrift] = []

    for col in baseline.columns:
        cs = next(x for x in schema if x.name == col)
        b = baseline[col]
        c = current[col]

        miss_d = dm.missingness_delta(b, c)

        if cs.kind == "numeric":
            psi = dm.psi_numeric(b, c)
            pval = dm.ks_pvalue(b, c)
            # score: higher PSI + higher missingness change + low p-value boost
            score = float(psi + abs(miss_d) + (0.2 if pval < 0.05 else 0.0))
            details = {"psi": psi, "ks_pvalue": pval}
        elif cs.kind == "categorical":
            cat = dm.categorical_shift(b, c)
            jsd = float(cat["js_divergence"])
            pval = float(cat["chi2_pvalue"])
            score = float(jsd + abs(miss_d) + (0.2 if pval < 0.05 else 0.0))
            details = cat
        else:
            # unknown/datetime: only missingness for now
            score = float(abs(miss_d))
            details = {}

        drifts.append(FeatureDrift(
            name=col,
            kind=cs.kind,
            missing_delta=miss_d,
            score=score,
            details=details
        ))

    drifts.sort(key=lambda d: d.score, reverse=True)
    return schema, drifts
