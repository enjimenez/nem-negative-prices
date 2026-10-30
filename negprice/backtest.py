"""Rolling-origin backtest of day-ahead negative-price forecasts.

Folds: a sliding 24-month training window that ends 2 days before each 1-month
test block. Settings and the yes/no cutoff are chosen on the last 3 months of the
training window (inner training part ends 2 days before validation starts), then
the chosen setting is refit on the full window and forecasts the test block.

Scored rows exclude administered-price days and half-hours without an AEMO 13:00-run
forecast, so every model and benchmark is scored on exactly the same rows.

Model interface (any object with):
    name: str
    candidates: list[dict]       settings to compare on the validation months
    fixed_cutoff: float | None   None = choose the max-F1 cutoff on validation
    fit(train, params) -> fitted
    predict(fitted, rows) -> np.ndarray of P(negative); rows never contain is_neg
"""

from dataclasses import dataclass
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import typer

from negprice.config import PROCESSED_DATA_DIR, REGIONS, REPORTS_DIR

app = typer.Typer()

TEST_START = pd.Timestamp("2023-10-01")
N_FOLDS = 36
TRAIN_MONTHS = 24
VAL_MONTHS = 3
GAP = pd.Timedelta(days=2)  # last training day = first forecast day - 2
BASELINE_WEEKS = 8
RECAL_CLIP = (-50.0, 50.0)
CUTOFF_GRID = np.round(np.arange(0.01, 1.0, 0.01), 2)
N_BOOT = 10_000
SEED = 0
LAST_12_START = pd.Timestamp("2025-10-01")

BLOCKS = ["night", "morning", "midday", "evening"]
SEASONS = {
    12: "summer", 1: "summer", 2: "summer",
    3: "autumn", 4: "autumn", 5: "autumn",
    6: "winter", 7: "winter", 8: "winter",
    9: "spring", 10: "spring", 11: "spring",
}  # fmt: skip


# ---------------------------------------------------------------- groupings


def hour_block(local_hour: pd.Series) -> pd.Series:
    """Local-time blocks on interval-end labels: (6, 10] morning, (10, 16] midday,
    (16, 21] evening, everything else night."""
    h = local_hour.to_numpy()
    block = np.select(
        [(h > 6) & (h <= 10), (h > 10) & (h <= 16), (h > 16) & (h <= 21)],
        ["morning", "midday", "evening"],
        default="night",
    )
    return pd.Series(block, index=local_hour.index)


def year_label(day: pd.Series) -> pd.Series:
    """12-month periods Oct-Sep, labelled e.g. '2023-24'."""
    start = day.dt.year - (day.dt.month < 10)
    return start.astype(str) + "-" + ((start + 1) % 100).map("{:02d}".format)


def scorable(df: pd.DataFrame) -> pd.Series:
    """Rows that count in any score: market prices and an AEMO forecast exist."""
    return ~df["admin_price"].astype(bool) & df["pd_rrp"].notna()


# -------------------------------------------------------------------- folds


@dataclass(frozen=True)
class Fold:
    number: int
    train_start: pd.Timestamp
    inner_end: pd.Timestamp  # last day used to fit candidates (val_start - 2)
    val_start: pd.Timestamp
    train_end: pd.Timestamp  # last training and validation day (test_start - 2)
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_folds(test_start: pd.Timestamp = TEST_START, n_folds: int = N_FOLDS) -> list[Fold]:
    folds = []
    for i in range(n_folds):
        d0 = test_start + pd.DateOffset(months=i)
        val_start = d0 - pd.DateOffset(months=VAL_MONTHS)
        folds.append(
            Fold(
                number=i + 1,
                train_start=d0 - pd.DateOffset(months=TRAIN_MONTHS),
                inner_end=val_start - GAP,
                val_start=val_start,
                train_end=d0 - GAP,
                test_start=d0,
                test_end=d0 + pd.DateOffset(months=1) - pd.Timedelta(days=1),
            )
        )
    return folds


# ------------------------------------------------------------------ baseline


def add_baseline(df: pd.DataFrame) -> pd.Series:
    """Share of negatives at the same half-hour on the same weekday over the previous
    8 weeks (D-7 ... D-56). NaN if any of those 8 days is missing or an admin day."""
    slot = ((df["HH_END"] - df["DAY"]) / pd.Timedelta(minutes=30)).round().astype(int)
    y = df["is_neg"].astype(float).where(~df["admin_price"].astype(bool))
    out = pd.Series(np.nan, index=df.index)
    for region in df["REGIONID"].unique():
        m = df["REGIONID"] == region
        piv = pd.DataFrame({"DAY": df.loc[m, "DAY"], "slot": slot[m], "y": y[m]}).pivot(
            index="DAY", columns="slot", values="y"
        )
        piv = piv.reindex(pd.date_range(piv.index.min(), piv.index.max(), freq="D"))
        total = sum(piv.shift(7 * k) for k in range(1, BASELINE_WEEKS + 1))
        share = (total / BASELINE_WEEKS).stack(future_stack=True)
        idx = pd.MultiIndex.from_arrays([df.loc[m, "DAY"], slot[m]])
        out[m] = share.reindex(idx).to_numpy()
    return out


# ------------------------------------------------------------------- models


class Baseline:
    """Seasonal naive: reads the precomputed base_prob column (see add_baseline)."""

    name = "baseline"
    candidates = [{}]
    fixed_cutoff = None

    def fit(self, train, params):
        return None

    def predict(self, fitted, rows):
        return rows["base_prob"].to_numpy(float)


class AemoRaw:
    """AEMO's raw call: probability 1 if the 13:00-run forecast price < $0, else 0."""

    name = "aemo_raw"
    candidates = [{}]
    fixed_cutoff = 0.5

    def fit(self, train, params):
        return None

    def predict(self, fitted, rows):
        p = rows["pd_rrp"]
        return np.where(p.notna(), (p < 0).astype(float), np.nan)


class AemoRecal:
    """Logistic regression of is_neg on the AEMO forecast price clipped to
    [-$50, +$50], one curve per hour block."""

    name = "aemo_recal"
    candidates = [{}]
    fixed_cutoff = None

    @staticmethod
    def _x(rows):
        x = rows["pd_rrp"].clip(*RECAL_CLIP) / RECAL_CLIP[1]  # scaled to [-1, 1]
        return x.to_numpy().reshape(-1, 1)

    def fit(self, train, params):
        train = train[train["pd_rrp"].notna()]
        blocks = hour_block(train["local_hour"])
        fitted = {}
        for b in BLOCKS:
            t = train[blocks == b]
            if t.empty:
                continue
            y = t["is_neg"].to_numpy()
            if y.min() == y.max():  # one class only: constant probability
                fitted[b] = float(y[0])
                continue
            # Large C = practically no regularisation (plain logistic regression)
            fitted[b] = LogisticRegression(C=1e6, max_iter=1000).fit(self._x(t), y)
        return fitted

    def predict(self, fitted, rows):
        out = np.full(len(rows), np.nan)
        blocks = hour_block(rows["local_hour"]).to_numpy()
        has = rows["pd_rrp"].notna().to_numpy()
        for b, f in fitted.items():
            m = (blocks == b) & has
            if not m.any():
                continue
            out[m] = f if isinstance(f, float) else f.predict_proba(self._x(rows[m]))[:, 1]
        return out


BENCHMARKS = [Baseline(), AemoRaw(), AemoRecal()]


# ------------------------------------------------------------------ metrics


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def brier_skill_score(b: float, b_ref: float) -> float:
    """Share of the reference's Brier error removed (0.2 = 20% less error)."""
    return 1 - b / b_ref


def recall_precision_f1(y, call) -> tuple[float, float, float]:
    y = np.asarray(y).astype(bool)
    call = np.asarray(call).astype(bool)
    tp = (y & call).sum()
    recall = tp / y.sum() if y.sum() else np.nan
    precision = tp / call.sum() if call.sum() else np.nan
    f1 = 2 * tp / (y.sum() + call.sum()) if (y.sum() + call.sum()) else np.nan
    return float(recall), float(precision), float(f1)


def best_f1_cutoff(y, p, grid=CUTOFF_GRID) -> float:
    """Lowest cutoff on the grid that maximises F1 (call = p >= cutoff)."""
    p = np.asarray(p, float)
    f1s = [recall_precision_f1(y, p >= c)[2] for c in grid]
    return float(grid[int(np.nanargmax(f1s))])


def score(g: pd.DataFrame) -> dict:
    y, p = g["is_neg"].to_numpy(), g["prob"].to_numpy(float)
    recall, precision, f1 = recall_precision_f1(y, g["call"].to_numpy())
    auc = roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan
    return {
        "n": len(g),
        "share_neg": y.mean(),
        "mean_prob": p.mean(),
        "brier": brier(y, p),
        "auc": auc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


def score_table(preds: pd.DataFrame, by: list[str], reference: str = "aemo_recal"):
    rows = [{**dict(zip(by, key)), **score(g)} for key, g in preds.groupby(by)]
    table = pd.DataFrame(rows)
    if reference in table["model"].values:
        keys = [k for k in by if k != "model"]
        ref = table.loc[table["model"] == reference, keys + ["brier"]]
        table = table.merge(ref.rename(columns={"brier": "brier_ref"}), on=keys, how="left")
        table["bss_vs_recal"] = brier_skill_score(table["brier"], table["brier_ref"])
        table = table.drop(columns="brier_ref")
    return table


def bootstrap_ci(d, n_boot: int = N_BOOT, seed: int = SEED, level: float = 0.95):
    """Mean of d with a percentile bootstrap interval (resampling the values of d)."""
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(means, [a, 1 - a])
    return float(d.mean()), float(lo), float(hi)


def compare(preds: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Monthly paired differences d = Brier(reference) - Brier(model), per region.
    d > 0 means the model beat the reference that month."""
    se = preds.assign(
        month=preds["DAY"].dt.to_period("M"),
        se=(preds["prob"] - preds["is_neg"]) ** 2,
    )
    monthly = se.pivot_table(index=["REGIONID", "month"], columns="model", values="se")
    rows = []
    for region, w in monthly.groupby(level="REGIONID"):
        for model in monthly.columns.drop(reference):
            d = (w[reference] - w[model]).to_numpy()
            mean, lo, hi = bootstrap_ci(d)
            half = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
            rows.append(
                {
                    "REGIONID": region,
                    "model": model,
                    "reference": reference,
                    "months": len(d),
                    "wins": int((d > 0).sum()),
                    "mean_diff": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "gauss_low": mean - half,
                    "gauss_high": mean + half,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- the loop


@dataclass
class FoldResult:
    params: dict
    cutoff: float
    val_brier: float
    test_index: pd.Index
    prob: np.ndarray


def run_fold(data: pd.DataFrame, model, fold: Fold) -> FoldResult:
    """One region's rows in, forecasts for the fold's test month out.
    Test-month labels are removed before predict, so they can't be read."""
    day = data["DAY"]
    inner = data[(day >= fold.train_start) & (day <= fold.inner_end)]
    val = data[(day >= fold.val_start) & (day <= fold.train_end)]
    val = val[scorable(val)]
    train = data[(day >= fold.train_start) & (day <= fold.train_end)]
    test = data[(day >= fold.test_start) & (day <= fold.test_end)].drop(columns="is_neg")

    best = None
    for params in model.candidates:
        p_val = model.predict(model.fit(inner, params), val.drop(columns="is_neg"))
        if np.isnan(p_val).any():
            raise ValueError(f"{model.name}: missing validation forecasts, fold {fold.number}")
        b = brier(val["is_neg"], p_val)
        if best is None or b < best[0]:
            best = (b, params, p_val)
    val_brier, params, p_val = best

    if model.fixed_cutoff is not None:
        cutoff = model.fixed_cutoff
    else:
        cutoff = best_f1_cutoff(val["is_neg"], p_val)
    prob = model.predict(model.fit(train, params), test)
    return FoldResult(params, cutoff, val_brier, test.index, prob)


def run_backtest(features: pd.DataFrame, models, folds, regions=REGIONS):
    df = features.copy()
    if "base_prob" not in df:
        df["base_prob"] = add_baseline(df)
    preds, selections = [], []
    for region in regions:
        data = df[df["REGIONID"] == region]
        for model in models:
            for fold in folds:
                r = run_fold(data, model, fold)
                test = data.loc[r.test_index]
                preds.append(
                    pd.DataFrame(
                        {
                            "REGIONID": region,
                            "model": model.name,
                            "fold": fold.number,
                            "HH_END": test["HH_END"].to_numpy(),
                            "DAY": test["DAY"].to_numpy(),
                            "local_hour": test["local_hour"].to_numpy(),
                            "admin_price": test["admin_price"].to_numpy(),
                            "pd_rrp": test["pd_rrp"].to_numpy(),
                            "is_neg": test["is_neg"].to_numpy(),
                            "prob": r.prob,
                            "call": r.prob >= r.cutoff,
                            "scored": scorable(test).to_numpy(),
                        }
                    )
                )
                selections.append(
                    {
                        "REGIONID": region,
                        "model": model.name,
                        "fold": fold.number,
                        "test_start": fold.test_start,
                        "params": str(r.params),
                        "cutoff": r.cutoff,
                        "val_brier": r.val_brier,
                    }
                )
    preds = pd.concat(preds, ignore_index=True)
    preds["block"] = hour_block(preds["local_hour"])
    preds["season"] = preds["DAY"].dt.month.map(SEASONS)
    preds["test_year"] = year_label(preds["DAY"])
    missing = preds["scored"] & preds["prob"].isna()
    if missing.any():
        raise ValueError(f"Missing forecasts on {missing.sum()} scored rows")
    return preds, pd.DataFrame(selections)


def report(preds: pd.DataFrame) -> dict[str, pd.DataFrame]:
    s = preds[preds["scored"]]
    return {
        "headline": score_table(s, ["model", "REGIONID"]),
        "last_12_months": score_table(s[s["DAY"] >= LAST_12_START], ["model", "REGIONID"]),
        "by_year": score_table(s, ["model", "REGIONID", "test_year"]),
        "by_block": score_table(s, ["model", "REGIONID", "block"]),
        "by_block_season": score_table(s, ["model", "REGIONID", "block", "season"]),
        "vs_recal": compare(s, "aemo_recal"),
        "vs_raw": compare(s, "aemo_raw"),
    }


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "features.parquet",
    preds_path: Path = PROCESSED_DATA_DIR / "backtest_predictions.parquet",
    out_dir: Path = REPORTS_DIR / "backtest",
):
    logger.info(f"Reading {features_path}")
    features = pd.read_parquet(features_path)
    folds = make_folds()
    logger.info(f"{len(folds)} folds: {folds[0].test_start:%b %Y} - {folds[-1].test_end:%b %Y}")
    preds, selections = run_backtest(features, BENCHMARKS, folds)
    preds.to_parquet(preds_path, index=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    selections.to_csv(out_dir / "selections.csv", index=False)
    tables = report(preds)
    for name, table in tables.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    logger.success(f"Predictions: {preds.shape} -> {preds_path}")
    logger.success(f"Tables: {', '.join(tables)} -> {out_dir}")
    with pd.option_context("display.width", 140, "display.precision", 4):
        print(tables["headline"].to_string(index=False))


if __name__ == "__main__":
    app()