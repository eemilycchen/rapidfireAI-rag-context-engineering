"""Results tracker for RapidFire AI RAG experiments.

After ``run_evals()`` completes, call ``dump_results()`` to export metrics to
Markdown, CSV, and JSON under ``logs/``.

- **Fit mode:** uses ``experiment.get_runs_info()`` and ``get_results()``.
- **Evals mode:** pass ``eval_results=<return value of run_evals()>``; those
  getters are not available on ``Experiment`` in evals mode.
""" 

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


def _unwrap_rf_metric(v: Any) -> Any:
    """Flatten RapidFire metric dicts like {'value': x, 'is_algebraic': ...} to scalar."""
    if isinstance(v, dict) and "value" in v and len(v) <= 6:
        inner = v["value"]
        if isinstance(inner, dict):
            return json.dumps(inner, sort_keys=True)
        return inner
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True)
    return v


def eval_results_to_dataframe(eval_results: Dict[int, tuple]) -> pd.DataFrame:
    """Build a results table from the dict returned by Experiment.run_evals() (evals mode).

    Each value is ``(aggregated_batch_results, cumulative_metrics)``; ``cumulative_metrics``
    uses nested ``{'value': ...}`` entries (see RapidFire controller).
    """
    rows = []
    for pipeline_id, pair in eval_results.items():
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        _, metrics = pair
        if not isinstance(metrics, dict):
            continue
        row: Dict[str, Any] = {"pipeline_id": pipeline_id}
        for key, raw in metrics.items():
            row[key] = _unwrap_rf_metric(raw)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "run_id" not in df.columns and "pipeline_id" in df.columns:
        df["run_id"] = df["pipeline_id"]
    return df


def dump_results(experiment, config_labels=None, output_dir="logs", eval_results=None):
    """Export experiment results to markdown, CSV, and JSON.

    Args:
        experiment: RapidFire Experiment instance (after run_evals completes)
        config_labels: dict mapping run_id (int) -> description string
        output_dir: where to write output files (default: logs/)
        eval_results: **evals mode only** — pass the return value of ``experiment.run_evals()``.
            ``get_runs_info()`` / ``get_results()`` exist only for ``mode='fit'``; evals metrics
            are returned from ``run_evals()`` as a dict keyed by pipeline id.

    Returns:
        pd.DataFrame with merged runs_info + results (fit), or eval metrics rows (evals)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mode = getattr(experiment, "mode", "fit")
    if mode == "evals":
        if eval_results is None:
            raise ValueError(
                "In evals mode, pass eval_results=experiment.run_evals(...) return value. "
                "Experiment.get_runs_info() and get_results() are only supported in fit mode."
            )
        merged = eval_results_to_dataframe(eval_results)
        if config_labels and not merged.empty and "run_id" in merged.columns:

            def _lookup_label(x):
                try:
                    return config_labels.get(int(x), "")
                except (TypeError, ValueError):
                    return ""

            merged.insert(1, "label", merged["run_id"].map(_lookup_label))
    else:
        runs_info = experiment.get_runs_info()
        results = experiment.get_results()

        final = results.sort_values("step").groupby("run_id").tail(1).reset_index(drop=True)
        merged = runs_info.merge(final, on="run_id", how="left")

        if config_labels:
            merged.insert(1, "label", merged["run_id"].map(config_labels).fillna(""))

    if merged.empty:
        print("Warning: no result rows to export.")

    csv_path = output_dir / "experiment_results.csv"
    merged.to_csv(csv_path, index=False)

    json_path = output_dir / "experiment_results.json"
    serializable = merged.copy()
    for col in serializable.columns:
        serializable[col] = serializable[col].apply(
            lambda x: str(x) if not isinstance(x, (int, float, str, bool, type(None))) else x
        )
    serializable.to_json(json_path, orient="records", indent=2)

    md_path = output_dir / "experiment_results.md"
    _write_markdown(merged, md_path, config_labels)

    print(f"Results exported:")
    print(f"  {md_path}")
    print(f"  {csv_path}")
    print(f"  {json_path}")
    return merged


def _format_table(df):
    """Format a DataFrame as a markdown table string."""
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def _write_markdown(df, path, labels=None):
    """Write human-readable markdown summary of experiment results."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "# RapidFire AI RAG Experiment Results",
        f"Generated: {timestamp}",
        "",
    ]
    if df.empty or "run_id" not in df.columns:
        lines.append("_No result rows to summarize._")
        lines.append("")
        Path(path).write_text("\n".join(lines))
        return

    metric_keywords = [
        "f1", "precision", "recall", "retrieval",
        "correctness", "faithfulness", "completeness",
        "generation", "total", "judge",
    ]
    metric_cols = [
        c for c in df.columns
        if any(k in c.lower() for k in metric_keywords)
    ]

    id_cols = ["run_id"]
    if "label" in df.columns:
        id_cols.append("label")
    if "status" in df.columns:
        id_cols.append("status")

    lines.append("## Run Summary")
    lines.append("")
    summary_cols = id_cols + [c for c in ["completed_steps", "total_steps"] if c in df.columns]
    lines.append(_format_table(df[summary_cols]))
    lines.append("")

    if metric_cols:
        lines.append("## Metrics")
        lines.append("")
        metric_df = df[id_cols + metric_cols].copy()
        for col in metric_cols:
            metric_df[col] = metric_df[col].apply(
                lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else str(x)
            )
        lines.append(_format_table(metric_df))
        lines.append("")

    for score_col in ["Retrieval Score", "F1@5"]:
        if score_col in df.columns:
            best_idx = df[score_col].idxmax()
            best = df.loc[best_idx]
            tag = f" ({labels[best['run_id']]})" if labels and best["run_id"] in labels else ""
            val = best[score_col]
            val_str = f"{float(val):.4f}" if isinstance(val, (int, float)) else str(val)
            lines.append("## Best Retrieval Config")
            lines.append(f"Run {best['run_id']}{tag} — {score_col}: {val_str}")
            lines.append("")
            break

    for score_col in ["Generation Score (3 released)", "Correctness (pass rate)"]:
        if score_col in df.columns:
            best_idx = df[score_col].idxmax()
            best = df.loc[best_idx]
            tag = f" ({labels[best['run_id']]})" if labels and best["run_id"] in labels else ""
            val = best[score_col]
            val_str = f"{float(val):.4f}" if isinstance(val, (int, float)) else str(val)
            lines.append("## Best Generation Config")
            lines.append(f"Run {best['run_id']}{tag} — {score_col}: {val_str}")
            lines.append("")
            break

    Path(path).write_text("\n".join(lines))
