from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chisquare

def missingness_delta(b: pd.Series, c: pd.Series) -> float:
    return float(c.isna().mean() - b.isna().mean())

def psi_numeric(b: pd.Series, c: pd.Series, bins: int = 10, eps: float = 1e-6) -> float:
    b = pd.to_numeric(b, errors="coerce")
    c = pd.to_numeric(c, errors="coerce")
    b = b.dropna()
    c = c.dropna()
    if b.empty or c.empty:
        return 0.0

    # baseline quantile bin edges (robust)
    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(b.values, qs))
    if len(edges) < 3:
        return 0.0

    b_counts, _ = np.histogram(b.values, bins=edges)
    c_counts, _ = np.histogram(c.values, bins=edges)

    b_p = b_counts / max(1, b_counts.sum())
    c_p = c_counts / max(1, c_counts.sum())

    b_p = np.clip(b_p, eps, 1)
    c_p = np.clip(c_p, eps, 1)
    return float(np.sum((c_p - b_p) * np.log(c_p / b_p)))

def ks_pvalue(b: pd.Series, c: pd.Series) -> float:
    b = pd.to_numeric(b, errors="coerce").dropna()
    c = pd.to_numeric(c, errors="coerce").dropna()
    if b.empty or c.empty:
        return 1.0
    return float(ks_2samp(b.values, c.values).pvalue)

def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))

def categorical_shift(b: pd.Series, c: pd.Series, top_k: int = 12) -> dict:
    b = b.astype("object")
    c = c.astype("object")

    b_counts = b.value_counts(dropna=False)
    c_counts = c.value_counts(dropna=False)

    cats = sorted(set(b_counts.index.tolist()) | set(c_counts.index.tolist()), key=lambda x: str(x))
    b_vec = np.array([b_counts.get(cat, 0) for cat in cats], dtype=float)
    c_vec = np.array([c_counts.get(cat, 0) for cat in cats], dtype=float)

    # chi-square expects same total scale; use expected from baseline proportions
    if b_vec.sum() == 0 or c_vec.sum() == 0:
        pval = 1.0
    else:
        expected = (b_vec / b_vec.sum()) * c_vec.sum()
        # Avoid zeros in expected
        expected = np.clip(expected, 1e-6, None)
        pval = float(chisquare(f_obs=c_vec, f_exp=expected).pvalue)

    b_p = b_vec / max(1, b_vec.sum())
    c_p = c_vec / max(1, c_vec.sum())
    jsd = js_divergence(b_p, c_p)

    # top changed categories
    diffs = (c_p - b_p)
    idx = np.argsort(np.abs(diffs))[::-1][:top_k]
    top = [{"category": str(cats[i]), "baseline_pct": float(b_p[i]), "current_pct": float(c_p[i]), "delta": float(diffs[i])} for i in idx]

    return {"chi2_pvalue": pval, "js_divergence": jsd, "top_changes": top}
