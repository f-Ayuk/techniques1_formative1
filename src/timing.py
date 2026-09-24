# ============================================================
# src/timing.py
# ============================================================
import time, platform, psutil
from contextlib import contextmanager


def hardware_info():
    return {
        "platform":  platform.platform(),
        "processor": platform.processor(),
        "cpu_count": psutil.cpu_count(logical=False),
        "ram_gb":    round(psutil.virtual_memory().total / 1024**3, 2),
    }


@contextmanager
def timer():
    t0 = time.time()
    box = {"seconds": None}
    try:
        yield box
    finally:
        box["seconds"] = time.time() - t0