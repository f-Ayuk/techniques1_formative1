# ============================================================
# scripts/build_report.py
# Injects figures, tables, and scalars into the report template.
# Usage:
#     python scripts/build_report.py
# Outputs:
#     report/final_report.md
# ============================================================
from pathlib import Path
import re
import pandas as pd

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
ROOT         = Path(__file__).resolve().parent.parent
RES_DIR      = ROOT / "results"
FIG_DIR      = ROOT / "figures"
REPORT_DIR   = ROOT / "report"
TEMPLATE     = REPORT_DIR / "report_template.md"
OUTPUT       = REPORT_DIR / "final_report.md"

# ------------------------------------------------------------
# Load artefacts produced by the notebook
# ------------------------------------------------------------
def load_csv_safe(path):
    """Return a DataFrame or None if the file doesn't exist."""
    return pd.read_csv(path) if path.exists() else None

metrics_df  = load_csv_safe(RES_DIR / "metrics.csv")
timing_df   = load_csv_safe(RES_DIR / "timing.csv")
worst_df    = load_csv_safe(RES_DIR / "worst_windows.csv")
top_areas   = load_csv_safe(RES_DIR / "top_areas.csv")
exp_log     = load_csv_safe(RES_DIR / "experiment_summary.csv")

top3 = top_areas["square_id"].tolist() if top_areas is not None else [5161, 5059, 5259]
highest = top3[0]

# ------------------------------------------------------------
# Builders
# ------------------------------------------------------------
def build_memory_comparison_table():
    """Static table from the notebook (single-file benchmark)."""
    df = pd.DataFrame([
        {"Strategy": "Naive TXT (float64 / int64)",
         "Load time (s)": 33.58, "DataFrame mem (MB)": 295.6,
         "Disk size (MB)": 307.9},
        {"Strategy": "Optimized (float32 / int32 / usecols)",
         "Load time (s)": 0.5,  "DataFrame mem (MB)": 157.0,
         "Disk size (MB)": 307.9},
        {"Strategy": "Parquet (snappy)",
         "Load time (s)": 0.17, "DataFrame mem (MB)": 157.0,
         "Disk size (MB)": 61.9},
    ])
    return df.to_markdown(index=False)


def build_metrics_table(area_id):
    """Metrics table for one area."""
    if metrics_df is None:
        return "_Not available — rerun the notebook._"
    sub = metrics_df[metrics_df["area"] == area_id].copy()
    if sub.empty:
        return f"_No metrics found for area {area_id}._"
    sub = sub[["model", "MAE", "MAPE", "RMSE"]].round(4)
    sub.columns = ["Model", "MAE", "MAPE (%)", "RMSE"]
    return sub.to_markdown(index=False)


def build_timing_table():
    """Aggregate training / execution time (mean ± std) over areas."""
    if exp_log is None:
        return "_No experiment log found._"
    test = exp_log[exp_log["stage"] == "test"].copy()
    if test.empty:
        return "_No 'test' stage rows in experiment log._"

    grouped = (test.groupby("model")[["train_time_s", "exec_time_s"]]
                    .agg(["mean", "std"]).round(2))

    rows = []
    for model in grouped.index:
        train_m = grouped.loc[model, ("train_time_s", "mean")]
        train_s = grouped.loc[model, ("train_time_s", "std")]
        exec_m  = grouped.loc[model, ("exec_time_s",  "mean")]
        exec_s  = grouped.loc[model, ("exec_time_s",  "std")]
        rows.append({
            "Model": model,
            "Training time (s)": f"{train_m:.2f} ± {train_s:.2f}",
            "Execution time (s)": f"{exec_m:.2f} ± {exec_s:.2f}",
        })
    return pd.DataFrame(rows).to_markdown(index=False)


def build_worst_windows_table():
    if worst_df is None:
        return "_No worst-window report found._"
    df = worst_df[["area", "model", "start", "end", "window_mae"]].copy()
    df.columns = ["Area", "Model", "Start", "End", "1h MAE"]
    df["1h MAE"] = df["1h MAE"].round(2)
    return df.to_markdown(index=False)


# ------------------------------------------------------------
# Scalar substitutions
# ------------------------------------------------------------
def compute_scalars():
    """Values pulled from the notebook's printed output (hard-coded)."""
    vals = {
        "N_FILES":       "62",
        "DATA_SIZE_GB":  "19.38",
        "MEM_REDUCTION": "46.9%",
        "DISK_REDUCTION": "79.9%",
        "CACHE_SIZE_GB": "3.84",
        "SPEEDUP":       "70",
        "SKEW":          "4.27",
        "KURT":          "25.52",
        "TOP1_SHARE":    "11.0%",
        "TOP3_AREAS":    f"Squares {', '.join(str(a) for a in top3)}",
        "TOP1_AREAS":    str(top3[0]),
        "TOP2_AREAS":    str(top3[1]),
        "TOP3_AREAS_INDIVIDUAL": str(top3[2]),
        "REQUIRED_AREAS": "4159 and 4556",
        "MEAN_5161":     "1,426.98",
        "MEAN_4159":     "274.88",
        "STL_VAR":       "86.4%",
        "HARDWARE_PLATFORM": "Windows 11 (10.0.26200)",
        "HARDWARE_CPU":  "Intel64 Family 6 Model 154, 12 cores",
        "HARDWARE_RAM":  "15.73",
    }

    # Mean MAE per model, computed from metrics.csv
    if metrics_df is not None:
        means = metrics_df.groupby("model")["MAE"].mean().round(3)
        for model in ("LSTM", "SARIMA", "CNN"):
            vals[f"MEAN_MAE_{model}"] = f"{means.get(model, float('nan')):.3f}"
    else:
        vals["MEAN_MAE_LSTM"] = "75.977"
        vals["MEAN_MAE_SARIMA"] = "86.063"
        vals["MEAN_MAE_CNN"] = "235.308"

    return vals


# ------------------------------------------------------------
# Template substitution
# ------------------------------------------------------------
def substitute_scalars(text, values):
    for k, v in values.items():
        text = text.replace("{{" + k + "}}", str(v))
    return text


def substitute_tables(text):
    """Replace {{TABLE:name}} with markdown tables."""
    table_map = {
        "memory_comparison": build_memory_comparison_table(),
        "timing":            build_timing_table(),
        "worst_windows":     build_worst_windows_table(),
    }
    # Per-area metrics
    for sid in top3:
        table_map[f"metrics_{sid}"] = build_metrics_table(sid)

    def repl(match):
        name = match.group(1)
        return table_map.get(name, f"_Missing table: {name}_")

    return re.sub(r"\{\{TABLE:([^}]+)\}\}", repl, text)


def substitute_figures(text):
    """Replace {{FIGURE:path/to.png|Caption}} with markdown image syntax.
    Paths are rewritten to be relative to the report file location."""
    def repl(match):
        rel_path = match.group(1).strip()
        caption  = (match.group(2) or "").strip()

        abs_path = ROOT / rel_path
        if not abs_path.exists():
            return f"_Missing figure: {rel_path}_"

        # Reports live in report/, so paths must go up one level
        md_path = f"../{rel_path}"
        img_md  = f"![{caption}]({md_path})"
        return f"{img_md}\n\n*{caption}*" if caption else img_md

    return re.sub(r"\{\{FIGURE:([^|}]+)(?:\|([^}]+))?\}\}", repl, text)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Template not found: {TEMPLATE}")

    text = TEMPLATE.read_text(encoding="utf-8")
    text = substitute_scalars(text, compute_scalars())
    text = substitute_tables(text)
    text = substitute_figures(text)

    OUTPUT.write_text(text, encoding="utf-8")
    print(f"Report written → {OUTPUT}")
    print(f"  Tables substituted : {text.count('|') // 6}")
    print(f"  Missing items left : {text.count('_Missing')}")


if __name__ == "__main__":
    main()