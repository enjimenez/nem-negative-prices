"""Tests for the zero-shot Chronos-2 forecasts.

A. Timing: nothing published after 13:00 on D-1 reaches the model.
B. Shape and alignment: right half-hours, right day, right region, no shifts.
C. Probability conversion: known answers on made-up quantiles.
D. Forecast file: one probability per half-hour, nothing missing (skipped until the run exists).
E. Real model: same input -> same output (skipped if the model can't be loaded).

A-C use synthetic data and a stand-in model, so they run in seconds.
"""

import json

import numpy as np
import pandas as pd
import pytest

from negprice import chronos_fm as cf

DAY = pd.Timestamp("2024-03-10")
ISSUE = pd.Timestamp("2024-03-09 13:00")
LEVELS = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
H = pd.Timedelta


def synthetic(region="SA1", start="2024-01-01", end="2024-03-20", seed=0):
    """Price series and feature table shaped like panel/features.parquet for one region."""
    hh = pd.date_range(f"{start} 00:30", f"{end} 00:00", freq="30min", name="HH_END")
    rng = np.random.default_rng(seed)
    price = pd.Series(rng.normal(30, 60, len(hh)), index=hh, name="rrp_mean")
    day = (hh - cf.STEP).normalize()
    issue = day - H(hours=11)
    feat = pd.DataFrame(index=hh)
    for c in cf.OWN_PD + cf.NEIGHBOUR_PD + cf.OTHER:
        feat[c] = rng.normal(size=len(hh))
    feat[f"pd_rrp_{region.lower()}"] = np.nan
    feat["is_holiday"] = 0
    for c in ["issue_time", "pd_known_at", "pd_nb_known_at", "pd_ext_known_at"]:
        feat[c] = issue
    feat["pvf_known_at"] = issue - H("30min")
    feat["silo_d2_known_at"] = day - H(hours=15)  # 09:00 on D-1
    return price, feat


class FakePipe:
    """Stand-in for Chronos2Pipeline. Price quantiles are all -10 (sure negative) when
    (day of month + hour) is even, else all +10 (sure positive), so the right answer
    is known for every half-hour."""

    quantiles = LEVELS

    def predict_df(self, df, future_df=None, prediction_length=None, quantile_levels=None,
                   context_length=None, **kwargs):
        rows = []
        for item, g in df.groupby("item_id", sort=False):
            assert len(g) == cf.CONTEXT
            ts = pd.date_range(g["timestamp"].iloc[-1] + cf.STEP, periods=prediction_length, freq="30min")
            if future_df is not None:
                assert (future_df.loc[future_df.item_id == item, "timestamp"].to_numpy() == ts.to_numpy()).all()
            sign = np.where((ts.day + ts.hour) % 2 == 0, -10.0, 10.0)
            r = pd.DataFrame({"item_id": item, "timestamp": ts, "target_name": "target", "predictions": sign})
            for q in quantile_levels:
                r[str(q)] = sign
            rows.append(r)
        return pd.concat(rows, ignore_index=True)


# ---------- A. Timing ----------

@pytest.mark.parametrize("with_covs", [False, True])
def test_history_ends_at_13h_on_d_minus_1(with_covs):
    price, feat = synthetic()
    hist, _ = cf.build_inputs(price, feat, "SA1", DAY, with_covs)
    assert hist["timestamp"].iloc[-1] == ISSUE
    assert len(hist) == cf.CONTEXT == 2016
    assert (hist["timestamp"].diff().dropna() == cf.STEP).all()


@pytest.mark.parametrize("with_covs", [False, True])
def test_inputs_identical_when_later_data_is_garbage(with_covs):
    """Overwrite every price after 13:00 on D-1 and every feature row published after it
    with garbage: the model inputs must not change at all."""
    price, feat = synthetic()
    hist, fut = cf.build_inputs(price, feat, "SA1", DAY, with_covs)

    price2, feat2 = price.copy(), feat.copy()
    price2[price2.index > ISSUE] = 1e6
    late = feat2[cf.KNOWN_AT].max(axis=1) > ISSUE
    assert late.any()
    feat2.loc[late, cf.covariates("SA1")] = 1e6
    hist2, fut2 = cf.build_inputs(price2, feat2, "SA1", DAY, with_covs)

    pd.testing.assert_frame_equal(hist, hist2)
    if with_covs:
        pd.testing.assert_frame_equal(fut, fut2)


def test_late_covariate_raises():
    price, feat = synthetic()
    feat.loc[pd.Timestamp("2024-03-10 12:00"), "pvf_known_at"] = ISSUE + H("30min")
    with pytest.raises(ValueError, match="published"):
        cf.build_inputs(price, feat, "SA1", DAY, with_covs=True)


def test_missing_price_in_history_raises():
    price, feat = synthetic()
    price[pd.Timestamp("2024-03-01 10:00")] = np.nan
    with pytest.raises(ValueError, match="missing prices"):
        cf.build_inputs(price, feat, "SA1", DAY, with_covs=False)


# ---------- B. Shape and alignment ----------

@pytest.mark.parametrize("region", ["SA1", "VIC1", "NSW1"])
def test_covariate_list(region):
    cols = cf.covariates(region)
    assert len(cols) == len(set(cols)) == 17
    assert f"pd_rrp_{region.lower()}" not in cols
    assert not {"is_neg", "rrp_mean", "price_d2", "neg_share_4w"} & set(cols)
    assert not any(c.endswith("known_at") for c in cols)


def test_future_part_covers_13h30_d_minus_1_to_midnight_d():
    price, feat = synthetic()
    _, fut = cf.build_inputs(price, feat, "SA1", DAY, with_covs=True)
    assert len(fut) == cf.HORIZON == 70
    assert fut["timestamp"].iloc[0] == ISSUE + cf.STEP
    assert fut["timestamp"].iloc[-1] == DAY + H(days=1)


def test_values_are_not_shifted():
    price, feat = synthetic()
    hist, fut = cf.build_inputs(price, feat, "SA1", DAY, with_covs=True)
    cols = cf.covariates("SA1")
    np.testing.assert_array_equal(hist["target"], price.loc[hist["timestamp"]].to_numpy())
    np.testing.assert_array_equal(hist[cols], feat.loc[hist["timestamp"], cols].to_numpy())
    np.testing.assert_array_equal(fut[cols], feat.loc[fut["timestamp"], cols].to_numpy())
    assert list(hist.columns[3:]) == list(fut.columns[2:]) == cols


@pytest.mark.parametrize("with_covs", [False, True])
def test_output_lands_on_the_right_half_hours(with_covs):
    price, feat = synthetic()
    days = [DAY - H(days=1), DAY, DAY + H(days=1)]
    out = cf.forecast(FakePipe(), price, feat, "SA1", days, with_covs)

    assert len(out) == 48 * len(days)
    assert not out.duplicated(["REGIONID", "HH_END"]).any()
    assert (out["REGIONID"] == "SA1").all()
    assert ((out["HH_END"] > out["DAY"]) & (out["HH_END"] <= out["DAY"] + H(days=1))).all()
    ts = pd.DatetimeIndex(out["HH_END"])
    expected = np.where((ts.day + ts.hour) % 2 == 0, 0.99, 0.01)
    np.testing.assert_array_equal(out["p_neg"].to_numpy(), expected)


# ---------- C. Probability conversion ----------

def linear_quantiles(zero_at):
    """Quantiles that cross $0 exactly at level `zero_at` (straight line, $100 per unit level)."""
    return np.array([[(lv - zero_at) * 100 for lv in LEVELS]])


def test_all_quantiles_positive_gives_floor():
    assert cf.p_negative(np.array([np.linspace(10, 200, 13)]), LEVELS)[0] == 0.01


def test_all_quantiles_negative_gives_ceiling():
    assert cf.p_negative(np.array([np.linspace(-200, -10, 13)]), LEVELS)[0] == 0.99


@pytest.mark.parametrize("zero_at", [0.15, 0.5, 0.35, 0.92])
def test_zero_between_quantiles_interpolates(zero_at):
    assert cf.p_negative(linear_quantiles(zero_at), LEVELS)[0] == pytest.approx(zero_at)


def test_crossing_quantiles_are_sorted_first():
    q = linear_quantiles(0.35)
    shuffled = q[:, np.random.default_rng(1).permutation(13)]
    assert cf.p_negative(shuffled, LEVELS)[0] == pytest.approx(0.35)


def test_lower_prices_never_lower_the_probability():
    base = np.array([np.linspace(-50, 150, 13)])
    p = [cf.p_negative(base - shift, LEVELS)[0] for shift in range(0, 300, 10)]
    assert all(b >= a for a, b in zip(p, p[1:]))


def test_probabilities_stay_in_1_to_99_percent():
    q = np.random.default_rng(2).normal(0, 100, size=(1000, 13))
    p = cf.p_negative(q, LEVELS)
    assert p.min() >= 0.01 and p.max() <= 0.99


# ---------- D. Forecast file (after the full run) ----------

needs_run = pytest.mark.skipif(not cf.OUT_FILE.exists(), reason="run `python -m negprice.chronos_fm` first")


@pytest.fixture(scope="module")
def fc():
    return pd.read_parquet(cf.OUT_FILE)


@needs_run
def test_one_probability_per_half_hour(fc):
    n_days = (cf.LAST_DAY - cf.FIRST_DAY).days + 1
    assert len(fc) == len(cf.VERSIONS) * 3 * n_days * 48
    assert not fc.duplicated(["version", "REGIONID", "HH_END"]).any()
    assert fc["p_neg"].notna().all()
    assert fc["p_neg"].between(0.01, 0.99).all()
    assert fc["DAY"].min() == cf.FIRST_DAY and fc["DAY"].max() == cf.LAST_DAY


@needs_run
def test_every_feature_row_has_a_forecast(fc):
    feats = pd.read_parquet(cf.PROCESSED_DATA_DIR / "features.parquet", columns=["REGIONID", "HH_END", "DAY"])
    feats = feats[feats["DAY"].between(cf.FIRST_DAY, cf.LAST_DAY)]
    for version, g in fc.groupby("version"):
        merged = feats.merge(g, on=["REGIONID", "HH_END"], how="left", suffixes=("", "_fc"))
        assert merged["p_neg"].notna().all(), version
        assert (merged["DAY"] == merged["DAY_fc"]).all(), version


@needs_run
def test_run_recorded_model_and_versions():
    meta = json.loads(cf.META_FILE.read_text())
    assert meta["model_id"] == cf.MODEL_ID
    assert meta["model_revision"]
    assert meta["context"] == cf.CONTEXT and meta["horizon"] == cf.HORIZON


# ---------- E. Real model ----------

@pytest.fixture(scope="module")
def pipe():
    pytest.importorskip("chronos")
    try:
        return cf.load_pipeline()
    except Exception as e:  # offline and not cached
        pytest.skip(f"model not available: {e}")


@pytest.mark.parametrize("with_covs", [False, True])
def test_real_model_same_input_same_output(pipe, with_covs):
    price, feat = synthetic()
    a = cf.forecast(pipe, price, feat, "SA1", [DAY], with_covs)
    b = cf.forecast(pipe, price, feat, "SA1", [DAY], with_covs)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("with_covs", [False, True])
def test_real_model_day_does_not_depend_on_its_batch(pipe, with_covs):
    """A day forecast alone must match the same day forecast inside a batch: the model
    must not share information across days (no cross-learning)."""
    price, feat = synthetic()
    alone = cf.forecast(pipe, price, feat, "SA1", [DAY], with_covs)
    batch = cf.forecast(pipe, price, feat, "SA1", [DAY - H(days=2), DAY, DAY + H(days=3)], with_covs)
    batch = batch[batch["DAY"] == DAY].reset_index(drop=True)
    np.testing.assert_allclose(alone["p_neg"], batch["p_neg"], atol=1e-4)