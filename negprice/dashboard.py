"""Export the tables behind the Power BI dashboard.

Heatmap: one row per scored half-hour of xgb_v3.
Forecast vs actual: the 14-day window in the middle of each season (8th to 21st of
Oct, Jan, Apr, Jul) of every test year, one row per half-hour per model, with AEMO
recal alongside so it is always on the chart.
Scorecard: BSS vs AEMO recal with 95% bootstrap intervals (bar chart), and every
metric per region x hour block x season x test year, each with "All" (matrix).
"""

from itertools import product
import shutil
from pathlib import Path

import pandas as pd
import numpy as np

from negprice.backtest import bss_intervals, score_table
from negprice.config import PROCESSED_DATA_DIR

MODEL = "xgb_v3"
REFERENCE = "aemo_recal"
# Scorecard and forecast-vs-actual models, in display order
MODEL_LABELS = {
    "xgb_v3": "XGBoost",
    "chronos_cov": "Chronos-2 with covariates",
    "chronos_price": "Chronos-2 price only",
    "baseline": "Seasonal-naive baseline",
    "aemo_recal": "AEMO recalibrated",
}
OTHER_MODELS = [m for m in MODEL_LABELS if m != REFERENCE]

WIN_DIR = Path("/media/esteban/24628C12628BE6C0/Users/esteb/negprice_dashboard")

OUT_DIR = PROCESSED_DATA_DIR / "dashboard"
SEASON = {12: "Summer", 1: "Summer", 2: "Summer",
          3: "Autumn", 4: "Autumn", 5: "Autumn",
          6: "Winter", 7: "Winter", 8: "Winter",
          9: "Spring", 10: "Spring", 11: "Spring"}
SEASON_ORDER = {"Summer": 1, "Autumn": 2, "Winter": 3, "Spring": 4}
BLOCK_ORDER = {"night": 1, "morning": 2, "midday": 3, "evening": 4}

WINDOW_MONTHS = {1: "Summer", 4: "Autumn", 7: "Winter", 10: "Spring"}  # middle months
WINDOW_DAYS = (8, 21)  # inclusive
N_WINDOWS = 12  # 4 seasons x 3 test years
MIN_NEG = 30  # slices with fewer negative half-hours get blank metrics


def load_scored() -> pd.DataFrame:
    preds = pd.read_parquet(PROCESSED_DATA_DIR / "backtest_predictions.parquet")
    need = {"model", "REGIONID", "HH_END", "DAY", "prob", "is_neg", "call", "scored",
            "block", "season", "test_year"}
    missing = need - set(preds.columns)
    assert not missing, f"Missing {missing}. Columns are: {list(preds.columns)}"
    return preds[preds["scored"].astype(bool)]


def nem_start(hh_end: pd.Series) -> pd.Series:
    """Interval start in NEM time (UTC+10), without a time zone, for Power BI."""
    if hh_end.dt.tz is not None:
        hh_end = hh_end.dt.tz_convert("Etc/GMT-10").dt.tz_localize(None)
    return hh_end - pd.Timedelta(minutes=30)


def region_label(regionid: pd.Series) -> pd.Series:
    return regionid.astype(str).str.rstrip("1")


# ------------------------------------------------------------------- heatmap


def build(preds: pd.DataFrame | None = None) -> pd.DataFrame:
    preds = load_scored() if preds is None else preds
    df = preds.loc[preds["model"] == MODEL, ["REGIONID", "HH_END", "prob", "is_neg"]]
    start = nem_start(df["HH_END"])

    out = pd.DataFrame({
        "region": region_label(df["REGIONID"]),
        "hh_start": start,
        "half_hour": start.dt.strftime("%H:%M"),
        "hh_order": start.dt.hour * 2 + start.dt.minute // 30 + 1,
        "season": start.dt.month.map(SEASON),
        "prob": df["prob"].astype(float),
        "is_neg": df["is_neg"].astype(int),
    })
    out["season_order"] = out["season"].map(SEASON_ORDER)

    # Same sample as the backtest table
    assert out["prob"].between(0, 1).all()
    assert out["is_neg"].isin([0, 1]).all()
    assert not out.duplicated(["region", "hh_start"]).any()
    assert out.groupby("region").size().eq(52_608).all(), out.groupby("region").size()
    assert out.groupby(["region", "hh_order"]).size().eq(1_096).all()
    return out


def check_table(out: pd.DataFrame) -> pd.DataFrame:
    """Same aggregates Power BI should show, for the numbers check."""
    keys = ["region", "season", "half_hour"]
    agg = dict(n=("is_neg", "size"), actual=("is_neg", "mean"), forecast=("prob", "mean"))
    t = pd.concat([
        out.groupby(keys).agg(**agg).reset_index(),
        out.assign(season="All").groupby(keys).agg(**agg).reset_index(),
    ])
    t["diff"] = t["forecast"] - t["actual"]
    return t


# -------------------------------------------------------- forecast vs actual


def forecast_vs_actual(preds: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per half-hour x region x model (not AEMO recal) in the 12 windows, with
    AEMO recal's probability in its own column. `DAY` is the NEM-time date of the
    interval start, so a window holds whole days 00:00-23:30."""
    preds = load_scored() if preds is None else preds
    day = preds["DAY"]
    in_window = day.dt.month.isin(list(WINDOW_MONTHS)) & day.dt.day.between(*WINDOW_DAYS)
    w = preds.loc[in_window & preds["model"].isin(MODEL_LABELS)]

    keys = ["REGIONID", "HH_END"]
    ref = w.loc[w["model"] == REFERENCE, [*keys, "prob", "is_neg"]]
    ref = ref.rename(columns={"prob": "prob_aemo_recal", "is_neg": "is_neg_ref"})
    df = w.loc[w["model"] != REFERENCE].merge(ref, on=keys, how="left", validate="m:1")
    assert df["prob_aemo_recal"].notna().all(), "AEMO recal missing on some rows"
    assert (df["is_neg"] == df["is_neg_ref"]).all()

    start = nem_start(df["HH_END"])
    out = pd.DataFrame({
        "region": region_label(df["REGIONID"]),
        "model": df["model"].map(MODEL_LABELS),
        "model_order": df["model"].map(list(MODEL_LABELS).index) + 1,
        "hh_start": start,
        "test_year": df["test_year"],
        "season": start.dt.month.map(WINDOW_MONTHS),
        "prob_model": df["prob"].astype(float),
        "prob_aemo_recal": df["prob_aemo_recal"].astype(float),
        "is_neg": df["is_neg"].astype(int),
    })
    out["season_order"] = out["season"].map(SEASON_ORDER)
    out = out.sort_values(["region", "model_order", "hh_start"], ignore_index=True)

    # 12 windows x 14 days x 48 half-hours for every region and model
    windows = out.groupby(["region", "model"])[["test_year", "season"]].value_counts()
    assert len(out.groupby(["test_year", "season"])) == N_WINDOWS
    assert windows.eq(14 * 48).all(), windows[windows.ne(14 * 48)]
    assert not out.duplicated(["region", "model", "hh_start"]).any()
    return out


def fva_check(out: pd.DataFrame) -> pd.DataFrame:
    keys = ["region", "test_year", "season", "model"]
    return out.groupby(keys).agg(
        n=("is_neg", "size"), first=("hh_start", "min"), last=("hh_start", "max"),
        share_neg=("is_neg", "mean"), mean_prob_model=("prob_model", "mean"),
        mean_prob_aemo_recal=("prob_aemo_recal", "mean"),
    ).reset_index()


# ----------------------------------------------------------------- scorecard


def scorecard_bars(preds: pd.DataFrame | None = None) -> pd.DataFrame:
    """BSS vs AEMO recal with 95% bootstrap intervals, all test rows, per region."""
    preds = load_scored() if preds is None else preds
    t = bss_intervals(preds, OTHER_MODELS, reference=REFERENCE)
    t.insert(0, "region", region_label(t.pop("REGIONID")))
    t.insert(1, "model_order", t["model"].map(list(MODEL_LABELS).index) + 1)
    t["model"] = t["model"].map(MODEL_LABELS)
    t["reference"] = t["reference"].map(MODEL_LABELS)
    assert len(t) == 3 * len(OTHER_MODELS)
    return t.sort_values(["region", "model_order"], ignore_index=True)


def scorecard_matrix(preds: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every metric per model x region x hour block x season x test year, where each of
    block, season and test year can also be "All"."""
    preds = load_scored() if preds is None else preds
    s = preds[preds["model"].isin(MODEL_LABELS)]
    by = ["model", "REGIONID", "block", "season", "test_year"]
    parts = []
    for keep in product([True, False], repeat=3):
        g = s.copy()
        for col, k in zip(["block", "season", "test_year"], keep):
            if not k:
                g[col] = "All"
        parts.append(score_table(g, by, reference=REFERENCE))
    t = pd.concat(parts, ignore_index=True)

    # BSS is undefined where AEMO recal's Brier is 0 (no negatives, recal ~0): blank, not inf
    t["bss_vs_recal"] = t["bss_vs_recal"].replace([np.inf, -np.inf], np.nan)

    t["n_neg"] = (t["share_neg"] * t["n"]).round().astype(int)

    # Too few negatives for stable metrics: blank them (n and n_neg stay visible)
    few = t["n_neg"] < MIN_NEG
    t.loc[few, ["brier", "auc", "recall", "precision", "f1", "bss_vs_recal"]] = np.nan
    
    t.insert(0, "region", region_label(t.pop("REGIONID")))
    t.insert(1, "model_order", t["model"].map(list(MODEL_LABELS).index) + 1)
    t["model"] = t["model"].map(MODEL_LABELS)
    t["block_order"] = t["block"].map(BLOCK_ORDER).fillna(0).astype(int)
    t["season"] = t["season"].str.capitalize()
    t["season_order"] = t["season"].map(SEASON_ORDER).fillna(0).astype(int)
    t["block"] = t["block"].str.capitalize()

    n_combos = 5 * 5 * 4  # (4 blocks + All) x (4 seasons + All) x (3 years + All)
    assert len(t) == len(MODEL_LABELS) * 3 * n_combos, len(t)
    assert not t.duplicated(["region", "model", "block", "season", "test_year"]).any()
    return t.sort_values(["region", "block_order", "season_order", "test_year",
                          "model_order"], ignore_index=True)


# --------------------------------------------------------------------- main


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False, coerce_timestamps="ms", allow_truncated_timestamps=True)


if __name__ == "__main__":
    preds = load_scored()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WIN_DIR.mkdir(parents=True, exist_ok=True)

    heat = build(preds)
    fva = forecast_vs_actual(preds)
    bars = scorecard_bars(preds)
    matrix = scorecard_matrix(preds)

    files = {
        "heatmap_halfhours.parquet": heat,
        "forecast_vs_actual.parquet": fva,
        "scorecard_bars.csv": bars,
        "scorecard_matrix.csv": matrix,
    }
    for name, df in files.items():
        path = OUT_DIR / name
        if name.endswith(".parquet"):
            write_parquet(df, path)
        else:
            df.to_csv(path, index=False)
        shutil.copy2(path, WIN_DIR / name)
        print(f"{len(df):>9,} rows -> {path.name}")
    check_table(heat).to_csv(OUT_DIR / "heatmap_check.csv", index=False)
    fva_check(fva).to_csv(OUT_DIR / "forecast_vs_actual_check.csv", index=False)
    print(f"Copied to {WIN_DIR}")