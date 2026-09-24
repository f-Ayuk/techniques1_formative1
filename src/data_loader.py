# ============================================================
# src/data_loader.py
# ============================================================
import gc
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from .config import DATA_DIR, CACHE_DIR, MILAN_COLS, OPT_DTYPES


def find_files(data_dir=DATA_DIR):
    """Return sorted list of Milan daily .txt files."""
    files = sorted(Path(data_dir).glob("sms-call-internet-mi-*.txt"))
    return [f for f in files if f.is_file()]


def _detect_format(filepath):
    """Return (separator, has_header, column_names)."""
    with open(filepath) as fh:
        first = fh.readline().rstrip("\n")
    if "\t" not in first:
        raise ValueError(f"Not tab-separated: {filepath.name}")
    sep = "\t"
    try:
        float(first.split(sep)[0])
        has_header = False
    except ValueError:
        has_header = True
    return sep, has_header, MILAN_COLS


def load_daily(filepath, usecols=None):
    """Read one daily .txt with Parquet cache. Returns DataFrame with
    `datetime` as a column."""
    filepath = Path(filepath)
    pq = CACHE_DIR / (filepath.stem + ".parquet")

    if pq.exists():
        df = pd.read_parquet(pq)
        return df[usecols] if usecols else df

    sep, has_header, cols = _detect_format(filepath)
    raw_uc = ([("Time Interval" if c == "datetime" else c) for c in usecols]
                if usecols else None)

    df = pd.read_csv(
        filepath, sep=sep,
        header=0 if has_header else None,
        names=None if has_header else cols,
        usecols=raw_uc if raw_uc else list(OPT_DTYPES.keys()),
        dtype={k: v for k, v in OPT_DTYPES.items()
                if k in (raw_uc or list(OPT_DTYPES.keys()))},
        engine="python",
    )
    if "Time Interval" in df.columns:
        df["Time Interval"] = pd.to_datetime(df["Time Interval"], unit="ms")
        df = df.rename(columns={"Time Interval": "datetime"})
    return df


def pre_cache_all(files=None):
    """One-time Parquet caching of every daily file."""
    files = files or find_files()
    for f in tqdm(files, desc="Caching to Parquet"):
        pq = CACHE_DIR / (f.stem + ".parquet")
        if pq.exists():
            continue
        df = load_daily(f)
        df.to_parquet(pq, engine="pyarrow", compression="snappy",
                        index=False)
        del df
        gc.collect()


def build_area_series(square_id, files=None):
    """Full-period 10-min Internet-traffic series for one area."""
    files = files or find_files()
    cache_path = CACHE_DIR / f"ts_square_{square_id}.parquet"

    if cache_path.exists():
        return (pd.read_parquet(cache_path)
                    .set_index("datetime")["Internet traffic activity"]
                    .sort_index())

    parts = []
    for f in tqdm(files, desc=f"Square {square_id}"):
        df = load_daily(f, usecols=["datetime", "Square id",
                                    "Internet traffic activity"])
        df = df[df["Square id"] == square_id]
        if len(df):
            parts.append(df)
        del df
        gc.collect()

    full = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()

    ts = (full.sort_values("datetime")
                .set_index("datetime")["Internet traffic activity"]
                .groupby(level=0).sum().sort_index())
    ts.name = "Internet traffic activity"
    ts.reset_index().to_parquet(cache_path, index=False)
    return ts


def aggregate_total_traffic(files=None):
    """Sum Internet traffic per Square id across all daily files."""
    files = files or find_files()
    agg_path = CACHE_DIR / "total_traffic_per_area.parquet"
    if agg_path.exists():
        return (pd.read_parquet(agg_path)
                    .set_index("Square id")["total_traffic"]
                    .sort_values(ascending=False))

    from collections import defaultdict
    totals = defaultdict(float)
    for f in tqdm(files, desc="Aggregating per-area totals"):
        df = load_daily(f, usecols=["Square id",
                                    "Internet traffic activity"])
        df["Square id"] = df["Square id"].astype("int32")
        df["Internet traffic activity"] = df["Internet traffic activity"].astype("float32")
        grouped = df.groupby("Square id")["Internet traffic activity"].sum()
        for sid, val in grouped.items():
            totals[sid] += float(val)
        del df, grouped
        gc.collect()

    s = pd.Series(totals, name="total_traffic").sort_values(ascending=False)
    s.index.name = "Square id"
    s.reset_index().to_parquet(agg_path, index=False)
    return s