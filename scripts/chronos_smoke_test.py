"""Smoke test: Chronos-2 small, zero-shot, one region, on validation-only days (Jul-Sep 2023).

Checks that the inputs build correctly (context ends at 13:00 on D-1, 70 future half-hours,
every covariate known by 13:00 on D-1), converts quantiles to P(price < 0), and times
both versions so the full run can be estimated. Uses no test-month data.
"""

import time

import numpy as np
import pandas as pd
from chronos import Chronos2Pipeline

REGION = "SA1"
FIRST_DAY = pd.Timestamp("2023-08-14")  # target day D; Jul-Sep 2023 is validation-only
N_DAYS = 14  # batch of days, for a realistic timing
CTX = 2016  # 6 weeks of half-hours
H = 70  # 13:30 on D-1 ... 24:00 on D
HH = pd.Timedelta("30min")
N_FORECASTS_FULL = 1188 * 3  # days Jul 2023 - Sep 2026 x 3 regions, per version

NEIGHBOURS = {"SA1": ["pd_rrp_vic1", "pd_rrp_nsw1"],
              "VIC1": ["pd_rrp_sa1", "pd_rrp_nsw1"],
              "NSW1": ["pd_rrp_sa1", "pd_rrp_vic1"]}
COVS = (["pd_rrp", "pd_totaldemand", "pd_availablegeneration", "pd_netinterchange",
         "pd_demand_and_nonschedgen", "pd_uigf", "pd_ss_solar_uigf", "pd_ss_wind_uigf"]
        + NEIGHBOURS[REGION] + ["pd_rrp_qld1", "pd_rrp_tas1", "pvf_mw",
                                "silo_rain_d2", "silo_max_temp_d2", "silo_radiation_d2", "is_holiday"])
KNOWN_AT = ["issue_time", "pd_known_at", "pd_nb_known_at", "pd_ext_known_at",
            "pvf_known_at", "silo_d2_known_at"]

pan = pd.read_parquet("data/processed/panel.parquet", columns=["REGIONID", "HH_END", "rrp_mean"])
fea = pd.read_parquet("data/processed/features.parquet")
price = pan[pan.REGIONID == REGION].set_index("HH_END").sort_index()["rrp_mean"]
feat = fea[fea.REGIONID == REGION].set_index("HH_END").sort_index()


def build(day, with_covs):
    """Context and future frames for target day D, issued at 13:00 on D-1."""
    issue = day - pd.Timedelta(hours=11)  # 13:00 on D-1
    ctx = pd.date_range(end=issue, periods=CTX, freq="30min")
    fut = pd.date_range(start=issue + HH, periods=H, freq="30min")
    assert fut[-1] == day + pd.Timedelta(days=1), "future must end at 24:00 on D"
    item = f"{REGION}_{day.date()}"

    df = pd.DataFrame({"item_id": item, "timestamp": ctx,
                       "target": price.reindex(ctx).to_numpy()})
    assert df["target"].notna().all(), f"missing prices in context for {day.date()}"
    if not with_covs:
        return df, None

    past = feat.reindex(ctx)
    future = feat.reindex(fut)
    known = pd.concat([past[KNOWN_AT], future[KNOWN_AT]]).max().max()
    assert known <= issue, f"a covariate is known after the issue time ({known})"
    assert future[COVS].notna().all().all(), "missing covariates in the future part"
    df = pd.concat([df, past[COVS].reset_index(drop=True)], axis=1)
    fdf = pd.concat([pd.DataFrame({"item_id": item, "timestamp": fut}),
                     future[COVS].reset_index(drop=True)], axis=1)
    return df, fdf


def p_negative(out, levels):
    """P(price < 0): interpolate the quantile level at $0; clipped to [1%, 99%] by np.interp."""
    qcols = [str(q) for q in levels]
    q = np.sort(out[qcols].to_numpy(), axis=1)  # sort in case quantiles cross
    return np.array([np.interp(0.0, row, levels) for row in q])


days = [FIRST_DAY + pd.Timedelta(days=i) for i in range(N_DAYS)]

t0 = time.perf_counter()
pipe = Chronos2Pipeline.from_pretrained("autogluon/chronos-2-small", device_map="cpu")
levels = list(pipe.quantiles)
print(f"Model loaded in {time.perf_counter() - t0:.1f} s; {len(levels)} quantile levels: {levels}")

results = {}
for name, with_covs in [("price_only", False), ("covariates", True)]:
    parts = [build(d, with_covs) for d in days]
    df = pd.concat([p[0] for p in parts], ignore_index=True)
    fdf = None if not with_covs else pd.concat([p[1] for p in parts], ignore_index=True)

    pipe.predict_df(df[df.item_id == df.item_id.iloc[0]],  # warm-up call, not timed
                    future_df=None if fdf is None else fdf[fdf.item_id == fdf.item_id.iloc[0]],
                    prediction_length=H, quantile_levels=levels, context_length=CTX)
    t0 = time.perf_counter()
    out = pipe.predict_df(df, future_df=fdf, prediction_length=H,
                          quantile_levels=levels, context_length=CTX)
    secs = time.perf_counter() - t0

    out["day"] = pd.to_datetime(out["item_id"].str.split("_").str[1])
    in_d = (out["timestamp"] > out["day"]) & (out["timestamp"] <= out["day"] + pd.Timedelta(days=1))
    out = out[in_d].copy()
    assert len(out) == 48 * N_DAYS, f"expected {48 * N_DAYS} rows of day D, got {len(out)}"
    out["p_neg"] = p_negative(out, levels)
    results[name] = out.set_index("timestamp")["p_neg"]

    per = secs / N_DAYS
    print(f"\n{name}: {secs:.1f} s for {N_DAYS} forecasts = {per:.2f} s each; "
          f"full run ({N_FORECASTS_FULL} forecasts) ~ {per * N_FORECASTS_FULL / 60:.0f} min")

check = pd.DataFrame(results)
check["actual_price"] = price.reindex(check.index)
check["is_neg"] = (check["actual_price"] < 0).astype(int)
print("\nP(negative) range:", {k: (round(v.min(), 3), round(v.max(), 3)) for k, v in results.items()})
print("\nFirst day, every 4th half-hour:")
print(check.iloc[:48:4].round(3).to_string())