# ============================================================
# src/config.py
# ============================================================
from pathlib import Path

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "milan_data"
CACHE_DIR = BASE_DIR / "cache"
FIG_DIR   = BASE_DIR / "figures"
RES_DIR   = BASE_DIR / "results"

EDA_FIG_DIR  = FIG_DIR / "eda"
FCST_FIG_DIR = FIG_DIR / "forecasts"
ERR_FIG_DIR  = FIG_DIR / "errors"
for d in (CACHE_DIR, EDA_FIG_DIR, FCST_FIG_DIR, ERR_FIG_DIR, RES_DIR):
    d.mkdir(parents=True, exist_ok=True)

MILAN_COLS = ["Square id", "Time Interval", "Country code",
                "SMS-in activity", "SMS-out activity",
                "Call-in activity", "Call-out activity",
                "Internet traffic activity"]

OPT_DTYPES = {
    "Square id": "int32",
    "Time Interval": "int64",
    "Country code": "int16",
    "SMS-in activity": "float32",
    "SMS-out activity": "float32",
    "Call-in activity": "float32",
    "Call-out activity": "float32",
    "Internet traffic activity": "float32",
}

# Split dates
TRAIN_END = "2013-12-09"
VAL_START = "2013-12-10"
VAL_END   = "2013-12-15"
TEST_START = "2013-12-16"
TEST_END   = "2013-12-22"

# Forecast setup
SEQ_LEN = 288          # 2 days of 10-min intervals
SEASONAL_PERIOD = 144  # 1 day
RANDOM_STATE = 42

# Report areas of interest
REQUIRED_AREAS = [4159, 4556]