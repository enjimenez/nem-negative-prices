"""Zero-shot Chronos-2 forecasts of P(price < 0) for each half-hour of day D.

Spec: Chronos-2 small, no training on our data, two versions:
  - price_only: the region's half-hourly price history only;
  - covariates: price history plus AEMO's 13:00 forecasts, rooftop PV forecast,
    SILO weather on D-2 and the holiday flag.
Forecast issued at 13:00 on D-1 with 6 weeks of history. Horizon 70 half-hours
(13:30 on D-1 ... 24:00 on D); only the 48 half-hours of day D are kept.
P(price < 0) = quantile level at $0, interpolated between the model's quantiles,
clipped to [1%, 99%]. Raw, no recalibration.

Forecasts don't depend on the backtest fold (nothing is fitted), so every day from
1 Jul 2023 (first validation month) to 30 Sep 2026 is forecast once and saved.

Run:  python -m negprice.chronos_fm
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from negprice.config import PROCESSED_DATA_DIR, REGIONS

MODEL_ID = "autogluon/chronos-2-small"
CONTEXT = 2016  # 6 weeks of half-hours
HORIZON = 70  # 13:30 on D-1 ... 24:00 on D
STEP = pd.Timedelta("30min")
ISSUE_BEFORE_DAY = pd.Timedelta(hours=11)  # 13:00 on D-1 = 00:00 on D minus 11 h
FIRST_DAY = pd.Timestamp("2023-07-01")
LAST_DAY = pd.Timestamp("2026-09-30")
VERSIONS = ("price_only", "covariates")
P_MIN, P_MAX = 0.01, 0.99
DAYS_PER_CALL = 28  # forecasts sent to the model at once (progress output, memory)

OUT_FILE = PROCESSED_DATA_DIR / "chronos_forecasts.parquet"
META_FILE = PROCESSED_DATA_DIR / "chronos_forecasts_meta.json"

OWN_PD = [
    "pd_rrp", "pd_totaldemand", "pd_availablegeneration", "pd_netinterchange",
    "pd_demand_and_nonschedgen", "pd_uigf", "pd_ss_solar_uigf", "pd_ss_wind_uigf",
]
NEIGHBOUR_PD = ["pd_rrp_sa1", "pd_rrp_vic1", "pd_rrp_nsw1"]
OTHER = [
    "pd_rrp_qld1", "pd_rrp_tas1", "pvf_mw",
    "silo_rain_d2", "silo_max_temp_d2", "silo_radiation_d2", "is_holiday",
]
KNOWN_AT = [
    "issue_time", "pd_known_at", "pd_nb_known_at", "pd_ext_known_at",
    "pvf_known_at", "silo_d2_known_at",
]


def covariates(region: str) -> list[str]:
    """The 17 covariates for a region (own-region neighbour column dropped)."""
    own = f"pd_rrp_{region.lower()}"
    return OWN_PD + [c for c in NEIGHBOUR_PD if c != own] + OTHER


def issue_time(day: pd.Timestamp) -> pd.Timestamp:
    return day - ISSUE_BEFORE_DAY


def item_name(region: str, day: pd.Timestamp) -> str:
    return f"{region}|{day:%Y-%m-%d}"


def build_inputs(
    price: pd.Series, feat: pd.DataFrame, region: str, day: pd.Timestamp, with_covs: bool
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Model inputs for target day D, using only information published by 13:00 on D-1.

    price: the region's rrp_mean indexed by HH_END.
    feat: the region's rows of features.parquet indexed by HH_END.
    Returns (history frame, future-covariates frame or None).
    """
    issue = issue_time(day)
    ctx = pd.date_range(end=issue, periods=CONTEXT, freq=STEP)
    fut = pd.date_range(start=issue + STEP, periods=HORIZON, freq=STEP)
    item = item_name(region, day)

    target = price.reindex(ctx)
    if target.isna().any():
        raise ValueError(f"{item}: {int(target.isna().sum())} missing prices in the history")
    hist = pd.DataFrame({"item_id": item, "timestamp": ctx, "target": target.to_numpy()})
    if not with_covs:
        return hist, None

    cols = covariates(region)
    past, future = feat.reindex(ctx), feat.reindex(fut)
    known = pd.concat([past[KNOWN_AT], future[KNOWN_AT]]).max().max()
    if known > issue:
        raise ValueError(f"{item}: a covariate was published at {known}, after {issue}")

    hist = pd.concat([hist, past[cols].reset_index(drop=True)], axis=1)
    fut_df = pd.concat(
        [pd.DataFrame({"item_id": item, "timestamp": fut}), future[cols].reset_index(drop=True)],
        axis=1,
    )
    return hist, fut_df


def p_negative(quantiles: np.ndarray, levels: list[float]) -> np.ndarray:
    """P(price < 0) per row: quantile level at $0, linear interpolation, clipped to [1%, 99%].

    quantiles: shape (n, len(levels)), prices at each quantile level. Rows are sorted
    first, so crossing quantiles can't break the interpolation.
    """
    q = np.sort(np.asarray(quantiles, dtype=float), axis=1)
    lv = np.asarray(levels, dtype=float)
    p = np.array([np.interp(0.0, row, lv) for row in q])
    return np.clip(p, P_MIN, P_MAX)


def forecast(pipe, price, feat, region, days, with_covs) -> pd.DataFrame:
    """P(price < 0) for each half-hour of each day in `days`: REGIONID, DAY, HH_END, p_neg."""
    levels = list(pipe.quantiles)
    parts = [build_inputs(price, feat, region, d, with_covs) for d in days]
    hist = pd.concat([p[0] for p in parts], ignore_index=True)
    fut = pd.concat([p[1] for p in parts], ignore_index=True) if with_covs else None

    out = pipe.predict_df(
        hist, future_df=fut, prediction_length=HORIZON,
        quantile_levels=levels, context_length=CONTEXT,
    )
    out["DAY"] = pd.to_datetime(out["item_id"].str.split("|").str[1])
    in_day = (out["timestamp"] > out["DAY"]) & (out["timestamp"] <= out["DAY"] + pd.Timedelta(days=1))
    out = out[in_day]
    if len(out) != 48 * len(days):
        raise ValueError(f"expected {48 * len(days)} half-hours of day D, got {len(out)}")

    return pd.DataFrame({
        "REGIONID": region,
        "DAY": out["DAY"].to_numpy(),
        "HH_END": out["timestamp"].to_numpy(),
        "p_neg": p_negative(out[[str(q) for q in levels]].to_numpy(), levels),
    })


def load_pipeline():
    from chronos import Chronos2Pipeline

    return Chronos2Pipeline.from_pretrained(MODEL_ID, device_map="cpu")


def model_revision() -> str | None:
    """Commit hash of the downloaded model snapshot, for the record."""
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(MODEL_ID, local_files_only=True).rstrip("/").split("/")[-1]
    except Exception:
        return None


def run() -> None:
    import chronos
    import torch

    panel = pd.read_parquet(PROCESSED_DATA_DIR / "panel.parquet", columns=["REGIONID", "HH_END", "rrp_mean"])
    features = pd.read_parquet(PROCESSED_DATA_DIR / "features.parquet")
    pipe = load_pipeline()
    days = list(pd.date_range(FIRST_DAY, LAST_DAY, freq="D"))
    chunks = [days[i:i + DAYS_PER_CALL] for i in range(0, len(days), DAYS_PER_CALL)]

    results = []
    for version in VERSIONS:
        for region in REGIONS:
            price = panel[panel.REGIONID == region].set_index("HH_END").sort_index()["rrp_mean"]
            feat = features[features.REGIONID == region].set_index("HH_END").sort_index()
            t0 = time.perf_counter()
            for i, chunk in enumerate(chunks, 1):
                df = forecast(pipe, price, feat, region, chunk, version == "covariates")
                results.append(df.assign(version=version))
                print(f"\r{version} {region}: {i}/{len(chunks)} chunks, "
                      f"{time.perf_counter() - t0:.0f} s", end="", flush=True)
            print()

    out = pd.concat(results, ignore_index=True)
    out.to_parquet(OUT_FILE, index=False)
    META_FILE.write_text(json.dumps({
        "model_id": MODEL_ID,
        "model_revision": model_revision(),
        "quantile_levels": list(pipe.quantiles),
        "context": CONTEXT,
        "horizon": HORIZON,
        "first_day": str(FIRST_DAY.date()),
        "last_day": str(LAST_DAY.date()),
        "chronos_version": chronos.__version__,
        "torch_version": torch.__version__,
    }, indent=2))
    print(f"Saved {len(out):,} rows to {OUT_FILE}")


if __name__ == "__main__":
    run()