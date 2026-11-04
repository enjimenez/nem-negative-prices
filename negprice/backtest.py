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

Each model's forecasts are cached in data/processed/backtest_cache/. A model is rerun
only when its fingerprint changes: the feature data, the fold and fitting code, the
model's own code and settings, or library versions (see fingerprint).
"""

from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import typer
from xgboost import XGBClassifier

import os
import subprocess
from datetime import datetime


from negprice.config import PROJ_ROOT, PROCESSED_DATA_DIR, REGIONS, REPORTS_DIR
from negprice.features import EXT_COLS, FEATURES

# Features of xgb_v1 and xgb_v2, kept as they were scored: everything except the
# QLD1/TAS1 forecast prices, which only xgb_v3 uses
BASE_FEATURES = [c for c in FEATURES if c not in EXT_COLS]

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
CACHE_DIR = PROCESSED_DATA_DIR / "backtest_cache"

BLOCKS = ["night", "morning", "midday", "evening"]
SEASONS = {
    12: "summer", 1: "summer", 2: "summer",
    3: "autumn", 4: "autumn", 5: "autumn",
    6: "winter", 7: "winter", 8: "winter",
    9: "spring", 10: "spring", 11: "spring",
}  # fmt: skip

# ---------------------------------------------------------------- run logging

RUN_LOG = REPORTS_DIR / "backtest" / "runs.csv"
CODE_PATHS = ["negprice", "tests", "Makefile", "pyproject.toml", "uv.lock"]

def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=PROJ_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def log_run(headline: pd.DataFrame, note: str, recomputed: list[str]) -> None:
    """Append the headline table of this run, with metadata, to the run log.
    `recomputed`: models fitted in this run (the others came from the cache)."""
    entry = headline.copy()
    entry.insert(0, "recomputed", " ".join(recomputed) or "none")
    entry.insert(0, "note", note)
    entry.insert(0, "code_changed_since_commit",
                 bool(_git("status", "--porcelain", "--", *CODE_PATHS)))
    entry.insert(0, "git_commit", _git("rev-parse", "--short", "HEAD"))
    entry.insert(0, "run_at", datetime.now().astimezone().isoformat(timespec="seconds"))
    if RUN_LOG.exists():
        entry = pd.concat([pd.read_csv(RUN_LOG), entry], ignore_index=True)
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry.to_csv(RUN_LOG, index=False)


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


XGB_V1_PARAMS = {
    "objective": "binary:logistic",
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "random_state": SEED,
    "n_jobs": -1,
}


class XGBoostV1:
    """Gradient-boosted trees on the known-at-issue features (decision L), one fixed
    setting. Missing values are left to XGBoost. The region's own neighbour-price
    column is dropped because it is always empty."""

    name = "xgb_v1"
    candidates = [XGB_V1_PARAMS]
    fixed_cutoff = None
    features = BASE_FEATURES

    @classmethod
    def feature_columns(cls, region: str) -> list[str]:
        return [c for c in cls.features if c != f"pd_rrp_{region.lower()}"]

    @staticmethod
    def _region(rows) -> str:
        regions = rows["REGIONID"].unique()
        if len(regions) != 1:
            raise ValueError(f"xgb_v1 expects one region per call, got {list(regions)}")
        return regions[0]

    def fit(self, train, params):
        cols = self.feature_columns(self._region(train))
        model = XGBClassifier(**params).fit(train[cols], train["is_neg"])
        return model, cols

    def predict(self, fitted, rows):
        model, cols = fitted
        return model.predict_proba(rows[cols])[:, 1]


XGB_V2_GRID = [
    {**XGB_V1_PARAMS, "max_depth": d, "n_estimators": n}
    for d in (4, 5, 6)
    for n in (50, 100, 150, 200)
]


class XGBoostV2(XGBoostV1):
    """xgb_v1 with a grid over tree depth and number of trees, chosen per fold on the
    validation months."""

    name = "xgb_v2"
    candidates = XGB_V2_GRID


class XGBoostV3(XGBoostV2):
    """xgb_v2 (same grid) plus the QLD1 and TAS1 forecast prices for every region."""

    name = "xgb_v3"
    features = FEATURES


MODELS = [*BENCHMARKS, XGBoostV1(), XGBoostV2(), XGBoostV3()]


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


def run_model(df: pd.DataFrame, model, folds, regions) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All folds of one model in every region: raw forecasts and chosen settings."""
    preds, selections = [], []
    for region in regions:
        data = df[df["REGIONID"] == region]
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
    return pd.concat(preds, ignore_index=True), pd.DataFrame(selections)


# ------------------------------------------------------------------- cache

# Code every model's forecasts depend on. Report and metric code is not listed: the
# tables are always rebuilt from the forecasts. A model that uses another module-level
# helper must add it here (or run with --force after changing it).
HARNESS_CODE = [make_folds, run_fold, run_model, scorable, add_baseline, hour_block,
                recall_precision_f1, best_f1_cutoff, brier]
HARNESS_SETTINGS = {"BASELINE_WEEKS": BASELINE_WEEKS, "RECAL_CLIP": RECAL_CLIP,
                    "CUTOFF_GRID": CUTOFF_GRID.tolist(), "BLOCKS": BLOCKS, "FEATURES": FEATURES}
LIBRARIES = ["numpy", "pandas", "scikit-learn", "xgboost"]


def data_hash(df: pd.DataFrame) -> str:
    """Hash of the values, column names and dtypes of the feature table."""
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    h.update(repr([(c, str(t)) for c, t in df.dtypes.items()]).encode())
    return h.hexdigest()


def fingerprint(model, features_hash: str, folds, regions) -> str:
    """Everything that decides a model's forecasts. Same fingerprint = same forecasts."""
    parts = {
        "features": features_hash,
        "folds": [repr(f) for f in folds],
        "regions": list(regions),
        "harness": [inspect.getsource(f) for f in HARNESS_CODE],
        "settings": HARNESS_SETTINGS,
        "model": [inspect.getsource(c) for c in type(model).__mro__ if c is not object],
        "candidates": repr(model.candidates),
        "fixed_cutoff": repr(model.fixed_cutoff),
        "libraries": {lib: version(lib) for lib in LIBRARIES},
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def load_cached(cache_dir: Path, name: str, fp: str):
    meta = cache_dir / f"{name}.json"
    if not meta.exists() or json.loads(meta.read_text())["fingerprint"] != fp:
        return None
    return (pd.read_parquet(cache_dir / f"{name}_preds.parquet"),
            pd.read_parquet(cache_dir / f"{name}_selections.parquet"))


def save_cached(cache_dir: Path, name: str, fp: str, preds, selections) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(cache_dir / f"{name}_preds.parquet", index=False)
    selections.to_parquet(cache_dir / f"{name}_selections.parquet", index=False)
    saved_at = datetime.now().astimezone().isoformat(timespec="seconds")
    meta = {"fingerprint": fp, "saved_at": saved_at}
    (cache_dir / f"{name}.json").write_text(json.dumps(meta))  # written last = complete


# ---------------------------------------------------------------- the run


def run_backtest(features: pd.DataFrame, models, folds, regions=REGIONS,
                 cache_dir: Path | None = None, force: bool = False):
    """Forecasts of every model. With cache_dir, a model whose fingerprint matches its
    cached run is loaded instead of refitted (force=True refits all). The selections
    table's `recomputed` column says which models were fitted in this call."""
    df = features.copy()
    if "base_prob" not in df:
        df["base_prob"] = add_baseline(df)
    features_hash = data_hash(features) if cache_dir is not None else None
    preds, selections = [], []
    for model in models:
        cached, fp = None, None
        if cache_dir is not None:
            fp = fingerprint(model, features_hash, folds, regions)
            cached = None if force else load_cached(cache_dir, model.name, fp)
        if cached is None:
            logger.info(f"{model.name}: fitting")
            p, s = run_model(df, model, folds, regions)
            if cache_dir is not None:
                save_cached(cache_dir, model.name, fp, p, s)
        else:
            logger.info(f"{model.name}: unchanged, loaded from cache")
            p, s = cached
        preds.append(p)
        selections.append(s.assign(recomputed=cached is None))
    preds = pd.concat(preds, ignore_index=True)
    preds["block"] = hour_block(preds["local_hour"])
    preds["season"] = preds["DAY"].dt.month.map(SEASONS)
    preds["test_year"] = year_label(preds["DAY"])
    missing = preds["scored"] & preds["prob"].isna()
    if missing.any():
        raise ValueError(f"Missing forecasts on {missing.sum()} scored rows")
    return preds, pd.concat(selections, ignore_index=True)


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
    force: bool = typer.Option(False, "--force", help="Refit every model, ignore the cache"),
):
    logger.info(f"Reading {features_path}")
    features = pd.read_parquet(features_path)
    folds = make_folds()
    logger.info(f"{len(folds)} folds: {folds[0].test_start:%b %Y} - {folds[-1].test_end:%b %Y}")
    preds, selections = run_backtest(features, MODELS, folds, cache_dir=CACHE_DIR, force=force)
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
    recomputed = selections.loc[selections["recomputed"], "model"].unique().tolist()
    log_run(pd.read_csv(REPORTS_DIR / "backtest" / "headline.csv"),
            os.environ["NOTE"].strip(), recomputed)

if __name__ == "__main__":
    note = os.environ.get("NOTE", "").strip()
    if not note:
        raise SystemExit('Set a note: make backtest NOTE="what changed and why"')
    app()
    