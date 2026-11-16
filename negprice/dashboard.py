"""Export the half-hour table behind the P(negative) heatmap for Power BI."""

import shutil
from pathlib import Path

import pandas as pd

from negprice.config import PROCESSED_DATA_DIR

MODEL = "xgb_v3"

WIN_DIR = Path("/media/esteban/24628C12628BE6C0/Users/esteb/negprice_dashboard")

OUT_DIR = PROCESSED_DATA_DIR / "dashboard"
SEASON = {12: "Summer", 1: "Summer", 2: "Summer",
          3: "Autumn", 4: "Autumn", 5: "Autumn",
          6: "Winter", 7: "Winter", 8: "Winter",
          9: "Spring", 10: "Spring", 11: "Spring"}
SEASON_ORDER = {"Summer": 1, "Autumn": 2, "Winter": 3, "Spring": 4}


def build() -> pd.DataFrame:
    preds = pd.read_parquet(PROCESSED_DATA_DIR / "backtest_predictions.parquet")
    need = {"model", "REGIONID", "HH_END", "prob", "is_neg", "scored"}
    missing = need - set(preds.columns)
    assert not missing, f"Missing {missing}. Columns are: {list(preds.columns)}"

    df = preds.loc[(preds["model"] == MODEL) & preds["scored"].astype(bool),
                   ["REGIONID", "HH_END", "prob", "is_neg"]].copy()

    end = df["HH_END"]
    if end.dt.tz is not None:  # keep NEM time (UTC+10), drop tz for Power BI
        end = end.dt.tz_convert("Etc/GMT-10").dt.tz_localize(None)
    start = end - pd.Timedelta(minutes=30)

    out = pd.DataFrame({
        "region": df["REGIONID"].astype(str).str.rstrip("1"),
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


if __name__ == "__main__":
    out = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pq = OUT_DIR / "heatmap_halfhours.parquet"
    out.to_parquet(pq, index=False, coerce_timestamps="ms", allow_truncated_timestamps=True)
    check_table(out).to_csv(OUT_DIR / "heatmap_check.csv", index=False)
    WIN_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pq, WIN_DIR / pq.name)
    print(f"{len(out):,} rows -> {pq} and {WIN_DIR}")