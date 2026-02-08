from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import pandas as pd
from app.core.utils import write_json, now_iso

def export_report_json(out_path: str, baseline_path: str, current_path: str, schema, drifts, top_n: int = 25) -> str:
    payload = {
        "generated_at": now_iso(),
        "baseline_path": baseline_path,
        "current_path": current_path,
        "top_n": top_n,
        "schema": [asdict(s) for s in schema],
        "top_drifts": [asdict(d) for d in drifts[:top_n]],
    }
    write_json(out_path, payload)
    return out_path

def export_report_md(out_path: str, baseline_path: str, current_path: str, drifts, top_n: int = 25) -> str:
    lines = []
    lines.append(f"# Data Drift Report\n")
    lines.append(f"- Generated: {now_iso()}")
    lines.append(f"- Baseline: `{baseline_path}`")
    lines.append(f"- Current: `{current_path}`\n")
    lines.append(f"## Top {top_n} drifted features\n")
    lines.append("| Rank | Feature | Kind | Score | Missing Δ | Key metric |")
    lines.append("|---:|---|---|---:|---:|---|")
    for i, d in enumerate(drifts[:top_n], 1):
        key = ""
        if d.kind == "numeric":
            key = f"PSI={d.details.get('psi', 0):.4f}, KS p={d.details.get('ks_pvalue', 1):.3g}"
        elif d.kind == "categorical":
            key = f"JSD={d.details.get('js_divergence', 0):.4f}, χ² p={d.details.get('chi2_pvalue', 1):.3g}"
        lines.append(f"| {i} | {d.name} | {d.kind} | {d.score:.4f} | {d.missing_delta:.4f} | {key} |")

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
    return out_path
