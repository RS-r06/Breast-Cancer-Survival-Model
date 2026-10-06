"""The two cohorts, made comparable.

Development: the Rotterdam tumour bank, 2,982 women who had surgery for
primary breast cancer in Rotterdam between 1978 and 1993.
Validation: the German Breast Cancer Study Group 2 trial, 686 women with
node-positive breast cancer recruited 1984 to 1989.

Both come from the R survival package (Therneau), as used by Royston and
Altman (2013) to show how to validate a survival model. Copies are fetched
from the Rdatasets collection.

The outcome is recurrence-free survival: the time to cancer coming back or
death, whichever comes first.
"""

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = "https://vincentarelbundock.github.io/Rdatasets/csv/survival/{}.csv"
CACHE = Path("data")
DAYS_PER_YEAR = 365.25
HORIZON = 5.0  # years

PREDICTORS = ["age", "meno", "size_20_50", "size_over_50", "grade3",
              "log_nodes", "log_pgr", "log_er", "hormon"]
LABELS = {
    "age": "Age (per year)",
    "meno": "Postmenopausal",
    "size_20_50": "Tumour 20 to 50 mm (vs 20 or less)",
    "size_over_50": "Tumour over 50 mm (vs 20 or less)",
    "grade3": "Grade 3 (vs 1 or 2)",
    "log_nodes": "Positive nodes, log(n + 1)",
    "log_pgr": "Progesterone receptor, log(fmol/l + 1)",
    "log_er": "Oestrogen receptor, log(fmol/l + 1)",
    "hormon": "Hormone therapy",
}


def fetch(name):
    path = CACHE / f"{name}.csv"
    if not path.exists():
        CACHE.mkdir(exist_ok=True)
        urllib.request.urlretrieve(SOURCE.format(name), path)
    return pd.read_csv(path)


def _common(frame, size_mm=None, size_cat=None):
    out = pd.DataFrame(index=frame.index)
    out["age"] = frame["age"].astype(float)
    out["meno"] = frame["meno"].astype(int)
    if size_mm is not None:
        out["size_20_50"] = ((size_mm > 20) & (size_mm <= 50)).astype(int)
        out["size_over_50"] = (size_mm > 50).astype(int)
    else:
        out["size_20_50"] = (size_cat == "20-50").astype(int)
        out["size_over_50"] = (size_cat == ">50").astype(int)
    out["grade3"] = (frame["grade"] == 3).astype(int)  # Rotterdam has no grade 1, so 1 and 2 are merged
    out["nodes"] = frame["nodes"].astype(int)
    out["log_nodes"] = np.log1p(frame["nodes"])
    out["log_pgr"] = np.log1p(frame["pgr"])
    out["log_er"] = np.log1p(frame["er"])
    out["hormon"] = frame["hormon"].astype(int)
    return out


def truncate(time, event, horizon=HORIZON):
    """Administrative censoring at the horizon: anything after it is unseen."""
    late = time > horizon
    return np.where(late, horizon, time), np.where(late, 0, event)


def rotterdam(node_positive_only=True):
    raw = fetch("rotterdam")
    data = _common(raw, size_cat=raw["size"])
    # Recurrence-free survival is the earlier of recurrence and death. 43
    # women died after their recurrence follow-up had ended; their death
    # falls in a period when a recurrence could not have been seen, so they
    # are censored at the end of recurrence follow-up instead (the approach
    # in the survival package's documentation of these data).
    unseen = (raw["recur"] == 0) & (raw["death"] == 1) & (raw["rtime"] < raw["dtime"])
    time = np.where(unseen, raw["rtime"], np.minimum(raw["rtime"], raw["dtime"]))
    event = np.where(unseen, 0, np.maximum(raw["recur"], raw["death"]))
    data["time"], data["event"] = truncate(time / DAYS_PER_YEAR, event)
    data["year"] = raw["year"]
    if node_positive_only:
        data = data[data["nodes"] > 0]
    return data.reset_index(drop=True)


def gbsg():
    raw = fetch("gbsg")
    data = _common(raw, size_mm=raw["size"])
    data["time"], data["event"] = truncate(raw["rfstime"] / DAYS_PER_YEAR, raw["status"])
    return data.reset_index(drop=True)
