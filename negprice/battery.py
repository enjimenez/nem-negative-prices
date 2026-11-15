"""Battery value of the day-ahead forecasts: a 2-hour battery, two rules.

Battery: 1 MW / 2 MWh, 85% round-trip efficiency (applied on discharge), at most one
cycle a day, empty at the start of each day, full 0-100% state of charge, no
degradation cost. Every half-hour settles at `rrp_mean` (mean of the six 5-minute
prices). The whole of day D is planned at 13:00 on D-1.

Optimiser: the best one-cycle schedule for day D given a price signal (one charge
phase, then one discharge phase; 0.5 MWh per half-hour; 2 MWh; 85% efficiency). Fed
actual prices it is perfect foresight, the ceiling. Fed AEMO's 13:00 forecast price
(`pd_rrp`) it is the "aemo_price" benchmark, which uses no P(negative) at all.

Charging rule (both rules, every P(negative) strategy): the 4 half-hours of day D with
the highest P, among those before the evening block, with P >= the fold's cutoff (max
F1 on validation). Fewer qualifying half-hours = partial charge. Ties are broken by a
fixed random order (same order for every strategy).

Rule v1 (first rule tried):
  Discharge after the last charging half-hour, in the half-hours with the highest
  median `rrp_mean` at that half-hour of day over the fold's training window
  (train_start ... test_start - 2 days), same season as the test month.
  No half-hour reaches the cutoff = no trade that day.
Rule v2 (second rule tried, a switch):
  If a half-hour reaches the cutoff: charge by P as above, discharge after the last
  charging half-hour in the half-hours with the highest `pd_rrp`.
  If none does: run the optimiser on `pd_rrp`.
Rule v3 (third and last rule tried): v2, but on days where only 1-3 half-hours reach the
  cutoff the rest of the battery is topped up: the remaining window half-hours are tried
  from the lowest `pd_rrp` up, and each is added if it raises the day's profit forecast
  with `pd_rrp` (selling in the highest-`pd_rrp` half-hours after the last charge).
All rules sell 0.85 x the energy bought, 0.5 MWh per half-hour, best-ranked first.

Headline: value per strategy in A$/MW/year, the share of the ceiling captured, and
paired differences (xgb_v3 - aemo_recal under each rule; xgb_v3 under v2 and v3 -
aemo_price)
with 95% bootstrap intervals over months.

Run:  python -m negprice.battery
"""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from negprice.backtest import N_BOOT, SEASONS, SEED, bootstrap_ci, hour_block, make_folds
from negprice.config import PROCESSED_DATA_DIR, REGIONS, REPORTS_DIR

app = typer.Typer()

ENERGY_MWH = 2.0  # 1 MW x 2 h
SLOT_MWH = 0.5  # 1 MW for half an hour, at the grid connection
RTE = 0.85  # round-trip efficiency: energy sold = RTE x energy bought
CHARGE_SLOTS = int(ENERGY_MWH / SLOT_MWH)  # 4 half-hours fill the battery
SLOTS = 48
STRATEGIES = ["xgb_v3", "aemo_recal"]  # P(negative) forecasts run through each rule
RULES = ["v1", "v2", "v3"]
CEILING = "perfect_foresight"
AEMO_PRICE = "aemo_price"
COMPARISONS = [  # (strategy, reference): difference = strategy - reference
    ("v1/xgb_v3", "v1/aemo_recal"),
    ("v2/xgb_v3", "v2/aemo_recal"),
    ("v2/xgb_v3", AEMO_PRICE),
    ("v3/xgb_v3", "v3/aemo_recal"),
    ("v3/xgb_v3", AEMO_PRICE),
]
OUT_DIR = REPORTS_DIR / "battery"


# ------------------------------------------------------------------ helpers


def slot_of_day(hh_end: pd.Series, day: pd.Series) -> pd.Series:
    """Half-hour of day 1 ... 48 (interval-end labels, NEM time)."""
    return ((hh_end - day) / pd.Timedelta(minutes=30)).round().astype(int)


def charge_window(local_hour: np.ndarray) -> np.ndarray:
    """Half-hours of one day (in slot order) before the first evening-block half-hour."""
    evening = (hour_block(pd.Series(local_hour)) == "evening").to_numpy()
    first_evening = int(np.argmax(evening)) if evening.any() else len(local_hour)
    return np.arange(len(local_hour)) < first_evening


def fill(order: np.ndarray, energy: float) -> np.ndarray:
    """Spread `energy` over half-hours in the given order, SLOT_MWH each, the
    remainder in the next one. Returns MWh per half-hour (length SLOTS)."""
    out = np.zeros(SLOTS)
    left = energy
    for s in order:
        if left <= 1e-12:
            break
        out[s] = min(SLOT_MWH, left)
        left -= out[s]
    if left > 1e-9:
        raise ValueError(f"not enough half-hours to place {energy} MWh")
    return out


def day_value(price: np.ndarray, bought: np.ndarray, sold: np.ndarray) -> float:
    """A$ earned in one day: sales minus purchases (purchases at a negative price earn)."""
    return float((sold * price).sum() - (bought * price).sum())


# --------------------------------------------------------------- the rule


def charge_by_p(prob, cutoff, window, tie_key) -> np.ndarray:
    """MWh bought per half-hour: SLOT_MWH in each of the (up to) CHARGE_SLOTS window
    half-hours with the highest P among those with P >= cutoff."""
    prob = np.asarray(prob, float)
    eligible = np.flatnonzero(np.asarray(window) & (prob >= cutoff))
    # highest P first; ties by the fixed random key
    ranked = eligible[np.lexsort((tie_key[eligible], -prob[eligible]))]
    bought = np.zeros(SLOTS)
    bought[ranked[:CHARGE_SLOTS]] = SLOT_MWH
    return bought


def sell_after(bought: np.ndarray, rank_by) -> np.ndarray:
    """MWh sold per half-hour: RTE x energy bought, in the half-hours after the last
    charging half-hour with the highest `rank_by` (ties: earlier half-hour first)."""
    if not bought.sum():
        return np.zeros(SLOTS)
    after = np.arange(np.flatnonzero(bought).max() + 1, SLOTS)
    order = after[np.lexsort((after, -np.asarray(rank_by, float)[after]))]
    return fill(order, RTE * bought.sum())


def rule_v1_schedule(prob, cutoff, window, profile, tie_key):
    """One day of rule v1. All inputs are length-SLOTS arrays in slot order, except
    `cutoff` (float). `profile` = median price per half-hour of day used to rank
    discharge half-hours. Returns (bought, sold) in MWh per half-hour."""
    bought = charge_by_p(prob, cutoff, window, tie_key)
    return bought, sell_after(bought, profile)


def rule_v2_schedule(prob, cutoff, window, pd_rrp, tie_key):
    """One day of rule v2: charge by P and sell where AEMO forecasts the highest price;
    if no half-hour reaches the cutoff, the optimiser on AEMO's forecast price."""
    bought = charge_by_p(prob, cutoff, window, tie_key)
    if not bought.sum():
        return optimise(pd_rrp)
    return bought, sell_after(bought, pd_rrp)


def rule_v3_schedule(prob, cutoff, window, pd_rrp, tie_key):
    """One day of rule v3: v2, plus a top-up of the battery on partial days. The
    half-hours chosen by P are always kept; the other window half-hours are tried from
    the lowest `pd_rrp` up and each is added if it raises the forecast profit (the day's
    value computed with `pd_rrp` instead of actual prices)."""
    bought = charge_by_p(prob, cutoff, window, tie_key)
    if not bought.sum():
        return optimise(pd_rrp)
    pd_rrp = np.asarray(pd_rrp, float)
    best = day_value(pd_rrp, bought, sell_after(bought, pd_rrp))
    free = np.flatnonzero(np.asarray(window) & (bought == 0))
    for s in free[np.argsort(pd_rrp[free], kind="stable")]:
        if bought.sum() >= ENERGY_MWH - 1e-9:
            break
        trial = bought.copy()
        trial[s] = SLOT_MWH
        value = day_value(pd_rrp, trial, sell_after(trial, pd_rrp))
        if value > best + 1e-9:
            bought, best = trial, value
    return bought, sell_after(bought, pd_rrp)


# ------------------------------------------------------- perfect foresight


def _fill_cost(sorted_prices: np.ndarray, energy: float) -> float:
    """Price-weighted sum of `energy` placed on the first half-hours of sorted_prices."""
    full = int(energy // SLOT_MWH + 1e-9)
    rest = energy - full * SLOT_MWH
    total = SLOT_MWH * sorted_prices[:full].sum()
    if rest > 1e-12:
        total += rest * sorted_prices[full]
    return float(total)


def optimise(signal):
    """Best one-cycle schedule for a price signal: charge in half-hours 1 ... k, then
    discharge in k+1 ... 48, for the best split k and best energy E <= ENERGY_MWH.
    For a given k, profit is concave and piecewise linear in E, so the optimum is at a
    breakpoint. No profitable option = no trade. Returns (bought, sold) in MWh."""
    signal = np.asarray(signal, float)
    best = (0.0, None, None)  # not trading is always possible
    points = [SLOT_MWH * j for j in range(1, CHARGE_SLOTS + 1)]
    points += [SLOT_MWH * j / RTE for j in range(1, CHARGE_SLOTS + 1)]
    for k in range(1, SLOTS):
        e_max = min(ENERGY_MWH, SLOT_MWH * k, SLOT_MWH * (SLOTS - k) / RTE)
        cheap, dear = np.sort(signal[:k]), np.sort(signal[k:])[::-1]
        for e in sorted({min(p, e_max) for p in points}):
            v = _fill_cost(dear, RTE * e) - _fill_cost(cheap, e)
            if v > best[0] + 1e-9:
                best = (v, k, e)
    _, k, e = best
    if k is None:
        return np.zeros(SLOTS), np.zeros(SLOTS)
    bought = fill(np.argsort(signal[:k], kind="stable"), e)
    sold = fill(k + np.argsort(-signal[k:], kind="stable"), RTE * e)
    return bought, sold


def perfect_foresight(price: np.ndarray):
    """The ceiling: the optimiser fed actual prices. Returns (value, bought, sold)."""
    bought, sold = optimise(price)
    return day_value(np.asarray(price, float), bought, sold), bought, sold


# ------------------------------------------------------------ the profile


def discharge_profile(prices: pd.DataFrame, fold, season: str) -> np.ndarray:
    """Median `rrp_mean` per half-hour of day over the fold's training window
    (train_start ... train_end), days of the given season only, admin days excluded.
    `prices`: one region's rows with DAY, slot, rrp_mean, admin_price."""
    day = prices["DAY"]
    m = (day >= fold.train_start) & (day <= fold.train_end) & ~prices["admin_price"]
    m &= day.dt.month.map(SEASONS) == season
    med = prices[m].groupby("slot")["rrp_mean"].median()
    return med.reindex(range(1, SLOTS + 1)).to_numpy(float)


# ------------------------------------------------------------- the backtest


def battery_backtest(panel: pd.DataFrame, preds: pd.DataFrame, selections: pd.DataFrame,
                     folds=None, regions=REGIONS, strategies=STRATEGIES) -> pd.DataFrame:
    """Daily value of every strategy under each rule, of the AEMO-price optimiser and of
    perfect foresight. One row per region x day x strategy ("v1/xgb_v3", "v2/aemo_recal",
    "aemo_price", "perfect_foresight", ...), with MWh bought and sold and the number of
    charging half-hours."""
    folds = folds or make_folds()
    panel = panel.assign(slot=slot_of_day(panel["HH_END"], panel["DAY"]))
    preds = preds[preds["model"].isin(strategies)]
    preds = preds.assign(slot=slot_of_day(preds["HH_END"], preds["DAY"]))
    cutoffs = selections.set_index(["REGIONID", "model", "fold"])["cutoff"]
    rows = []
    for region in regions:
        prices = panel.loc[panel["REGIONID"] == region,
                           ["DAY", "HH_END", "slot", "rrp_mean", "admin_price"]]
        price_by_day = prices.set_index(["DAY", "slot"])["rrp_mean"]
        pr = preds[preds["REGIONID"] == region]
        rng = np.random.default_rng(SEED)
        for fold in folds:
            test = pr[(pr["DAY"] >= fold.test_start) & (pr["DAY"] <= fold.test_end)]
            if test.empty:
                continue
            if not test["scored"].all():
                raise ValueError(f"{region} fold {fold.number}: unscored test rows")
            season = SEASONS[fold.test_start.month]
            profile = discharge_profile(prices, fold, season)
            if np.isnan(profile).any():
                raise ValueError(f"{region} fold {fold.number}: profile has gaps")
            for day, g in test.groupby("DAY"):
                if g["model"].value_counts().ne(SLOTS).any():
                    raise ValueError(f"{region} {day:%Y-%m-%d}: not 48 half-hours")
                price = price_by_day.loc[day].reindex(range(1, SLOTS + 1)).to_numpy(float)
                tie_key = rng.random(SLOTS)  # same key for every strategy
                first = g[g["model"] == strategies[0]].sort_values("slot")
                pd_rrp = first["pd_rrp"].to_numpy(float)
                window = charge_window(first["local_hour"].to_numpy())
                out = [(CEILING, *optimise(price)), (AEMO_PRICE, *optimise(pd_rrp))]
                for model, gm in g.groupby("model"):
                    prob = gm.sort_values("slot")["prob"].to_numpy()
                    cutoff = cutoffs[(region, model, fold.number)]
                    out.append((f"v1/{model}",
                                *rule_v1_schedule(prob, cutoff, window, profile, tie_key)))
                    out.append((f"v2/{model}",
                                *rule_v2_schedule(prob, cutoff, window, pd_rrp, tie_key)))
                    out.append((f"v3/{model}",
                                *rule_v3_schedule(prob, cutoff, window, pd_rrp, tie_key)))
                for name, b, s in out:
                    rows.append({"REGIONID": region, "DAY": day, "fold": fold.number,
                                 "strategy": name, "value": day_value(price, b, s),
                                 "mwh_bought": b.sum(), "mwh_sold": s.sum(),
                                 "charge_slots": int((b > 0).sum())})
    daily = pd.DataFrame(rows)
    daily["month"] = daily["DAY"].dt.to_period("M")
    return daily


# ---------------------------------------------------------------- summary


def annualise(monthly_totals) -> float:
    """A$ per MW per year from monthly totals: mean month x 12."""
    return float(np.mean(monthly_totals) * 12)


def bootstrap_p(d, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """From the same resampled means as `bootstrap_ci` (same seed and draws): the share
    of resampled means above 0, and a two-sided p-value for "no difference",
    2 x the smaller tail (capped at 1). With 10,000 draws, 0 means p < 0.0001."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    above = float((means > 0).mean())
    return above, min(1.0, 2 * min(above, float((means <= 0).mean())))


def summarise(daily: pd.DataFrame, comparisons=COMPARISONS) -> tuple[pd.DataFrame,
                                                                     pd.DataFrame]:
    """Per region: (1) the ceiling and each strategy's value (A$/MW/year) and share of
    the ceiling captured; (2) paired monthly differences strategy - reference with a 95%
    bootstrap interval, the difference as % of the reference's value, the share of
    resampled means above 0, a two-sided bootstrap p-value and monthly wins."""
    monthly = daily.pivot_table(index=["REGIONID", "month"], columns="strategy",
                                values="value", aggfunc="sum")
    values, diffs = [], []
    for region, w in monthly.groupby(level="REGIONID"):
        ceiling = annualise(w[CEILING])
        for s in [c for c in w.columns if c != CEILING]:
            values.append({"REGIONID": region, "strategy": s, "months": len(w),
                           "ceiling_aud_mw_yr": ceiling, "value_aud_mw_yr": annualise(w[s]),
                           "share_of_ceiling": annualise(w[s]) / ceiling})
        for s, ref in comparisons:
            d = (w[s] - w[ref]).to_numpy()
            mean, lo, hi = bootstrap_ci(d)
            above, p = bootstrap_p(d)
            diffs.append({"REGIONID": region, "strategy": s, "reference": ref,
                          "months": len(d), "diff_aud_mw_yr": mean * 12, "ci_low": lo * 12,
                          "ci_high": hi * 12,
                          "diff_pct_of_reference": mean * 12 / annualise(w[ref]),
                          "share_resamples_above_0": above, "p_two_sided": p,
                          "wins": int((d > 0).sum())})
    return pd.DataFrame(values), pd.DataFrame(diffs)


@app.command()
def main(
    panel_path: Path = PROCESSED_DATA_DIR / "panel.parquet",
    preds_path: Path = PROCESSED_DATA_DIR / "backtest_predictions.parquet",
    selections_path: Path = REPORTS_DIR / "backtest" / "selections.csv",
    out_dir: Path = OUT_DIR,
):
    panel = pd.read_parquet(panel_path, columns=["REGIONID", "HH_END", "DAY", "rrp_mean",
                                                 "admin_price"])
    preds = pd.read_parquet(preds_path)
    selections = pd.read_csv(selections_path)
    daily = battery_backtest(panel, preds, selections)
    values, diffs = summarise(daily)
    out_dir.mkdir(parents=True, exist_ok=True)
    daily.assign(month=daily["month"].astype(str)).to_csv(out_dir / "daily.csv", index=False)
    values.to_csv(out_dir / "values.csv", index=False)
    diffs.to_csv(out_dir / "differences.csv", index=False)
    logger.success(f"Daily values {daily.shape}, values and differences -> {out_dir}")
    with pd.option_context("display.width", 160, "display.precision", 3):
        print(values.to_string(index=False), "\n")
        print(diffs.to_string(index=False))


if __name__ == "__main__":
    app()