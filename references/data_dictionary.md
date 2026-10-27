# Data dictionary: half-hourly panel

*Output of `negprice.processing.build_panel` + `add_admin_price_flag` + `add_target`. One row per region and half-hour. 262,944 rows = 3 regions × 1,826 days × 48 half-hours (1 Oct 2021 – 30 Sep 2026). Last updated 27 Oct 2026.*

## Conventions

- **Time:** NEM time (UTC+10 all year, no daylight saving). Timestamps are naive (no timezone attached), `datetime64[us]`.
- **Interval-end labels:** a half-hour labelled 12:00 covers 11:30–12:00. Verified: the centre of the rooftop PV curve tracks solar noon + 15 min all year, including the equation of time.
- **Day D:** the 48 half-hours ending D 00:30 … D 24:00. Not AEMO's trading day, which runs 04:00–04:00.
- **Units:** prices in $/MWh (A$, nominal); power in MW (half-hour average of the source values).
- **Leakage rule:** forecast issued at 13:00 on D-1. A value may be a model input only if its **known-at** time is ≤ 13:00 on D-1. Every source has a `*_known_at` column, or a fixed rule (below), for this.

## Prefixes

| Prefix | Source | What it is | Actual or forecast? |
|---|---|---|---|
| *(none)* | AEMO dispatch (`DISPATCHPRICE`, `DISPATCHREGIONSUM`), 5-min, `INTERVENTION = 0` | What actually happened in the market, averaged to half-hours | Actual |
| `pv_actual_` | AEMO `ROOFTOP_PV_ACTUAL`, `TYPE = MEASUREMENT` | AEMO's estimate of rooftop solar output | Actual (estimated) |
| `pd_` | AEMO **predispatch**, "13:00 run" on D-1 (`PREDISPATCHPRICE`, `PREDISPATCHREGIONSUM`) | AEMO's own day-ahead forecast of price, demand and supply: the dispatch engine run on the bids in place when bids closed at 12:30 on D-1 | Forecast |
| `pvf_` | AEMO `ROOFTOP_PV_FORECAST` (ASEFS2), latest version ≤ 12:30 on D-1 | AEMO's day-ahead forecast of rooftop solar output | Forecast |
| `silo_` | SILO DataDrill (gridded observed weather), one point per region | Observed daily weather | Actual |

Same root name, different prefix = same quantity, actual vs forecast. `totaldemand` is what demand turned out to be; `pd_totaldemand` is what AEMO forecast at 13:00 the day before.

**Role** column below:
- **Target:** what the models predict.
- **Lagged input:** usable only from half-hours already finished by 13:00 on D-1 (values for day D itself would leak).
- **Input:** known by 13:00 on D-1 for all of day D.
- **Benchmark:** AEMO's forecast that models must beat.
- **What-if only:** day-D actual weather, for the "perfect weather" ceiling model; never in the headline model.
- **QC:** quality control.

## Keys

| Column | Type | Description |
|---|---|---|
| `REGIONID` | str | AEMO region: `SA1`, `VIC1`, `NSW1` (NSW1 includes the ACT) |
| `HH_END` | datetime | End of the half-hour (interval-end label, NEM time) |
| `DAY` | datetime (date) | Calendar day D the half-hour belongs to: `(HH_END − 30 min).normalize()`. The half-hour ending 00:00 belongs to the previous day |

## Actual prices (`DISPATCHPRICE`)

Six 5-minute values per half-hour (the intervals ending HH:05 … HH:30 belong to the half-hour ending HH:30). Known at: end of each interval, so the half-hour is fully known at `HH_END`.

| Column | Type | Description | Role |
|---|---|---|---|
| `rrp_mean` | float, $/MWh | Mean of the six 5-min regional reference prices (RRP). Defines the target (`is_neg`). Also used for the battery dollar figure | Defines target; lagged input |
| `rrp_min` | float, $/MWh | Minimum of the six 5-min RRPs. `rrp_min < 0` ⇔ at least one 5-min interval negative. Not the target (see `is_neg`); kept for analysis | Lagged input |
| `n_neg_5min` | int, 0–6 | Number of 5-min intervals with RRP < 0. Duration view of negative prices within the half-hour; kept for analysis | Lagged input |
| `n_5min` | int | Number of 5-min intervals in the half-hour. Always 6 (checked) | QC |

Range: floor −$1,000; the maximum each financial year equals that year's market price cap (up to $20,300 in FY2026). About 0.5–0.7% of 5-min prices are exactly $0, so "< 0" vs "≤ 0" moves base rates by under 1 point.

## Target

| Column | Type | Description | Role |
|---|---|---|---|
| `is_neg` | int8, 0/1 | 1 if `rrp_mean` < 0 (half-hour mean of the six 5-min RRPs strictly below $0), else 0. A $0 half-hour counts as not negative. Defined on every row, including `admin_price` days, which are excluded at scoring. Share of 1s over the full window: SA 25.2%, VIC 21.3%, NSW 6.7% | Target |

Why the half-hour mean and not "any 5-min interval negative": AEMO's day-ahead forecast is half-hourly, so the benchmark compares like with like; a battery charging through the half-hour is paid the mean price; and single 5-min dips are mostly real-time volatility that can't be forecast a day ahead.

## Actual region summary (`DISPATCHREGIONSUM`)

Half-hour mean of the six 5-min values, MW. Known at `HH_END`.

| Column | Description | Role |
|---|---|---|
| `totaldemand` | Regional demand met by scheduled and semi-scheduled generation and net imports (AEMO's dispatch demand). Rooftop PV reduces it, so it **can be negative** (SA minimum −284 MW, Christmas Day 2025). Don't clip | Lagged input |
| `availablegeneration` | Total generation offered as available in the region by scheduled and semi-scheduled units | Lagged input |
| `netinterchange` | Net interconnector flow out of the region (positive = net export) | Lagged input |
| `totalintermittentgeneration` | AEMO's allowance for non-scheduled intermittent generation (small wind and solar farms not dispatched by AEMO) | Lagged input |
| `demand_and_nonschedgen` | Demand including the non-scheduled generation allowance: roughly `totaldemand + totalintermittentgeneration` | Lagged input |
| `uigf` | Unconstrained Intermittent Generation Forecast: what semi-scheduled wind and solar farms *could* produce if not curtailed (5-min-ahead availability). A measure of renewable supply available to the market | Lagged input |

## Rooftop PV actuals (`ROOFTOP_PV_ACTUAL`)

| Column | Type | Description | Role |
|---|---|---|---|
| `pv_actual_mw` | float, MW | AEMO's estimate of regional rooftop PV output for the half-hour (`TYPE = MEASUREMENT`: a sample of metered systems scaled up to installed capacity). 34 half-hours missing (not filled). 0 at night | Lagged input; scores `pvf_` |
| `pv_actual_known_at` | datetime | `LASTCHANGED`: publication time. 19–36 min after `HH_END`; no later revisions in the archive, so stored values = values known at the time | Leakage control |

## AEMO predispatch, 13:00 run (`pd_`)

All columns come from **one run per target day**: run 18 of the trading day, labelled 13:00 on D-1 and executed after bids closed at 12:30. Lead time 11.5 h (first half-hour of D) to 35 h (last). `INTERVENTION = 0`. **Missing for target days 2 Oct – 1 Nov 2022** (AEMO archive gap): 1,488 half-hours per region.

Predispatch is not a statistical forecast: it runs the dispatch engine on the bids in place at 12:30 on D-1, using AEMO's demand forecast (DFS) and renewable forecasts (ASEFS2/AWEFS). Participants rebid volumes afterwards, so forecast and outcome can differ systematically (NSW over-forecasts negative prices).

| Column | Type | Description | Role |
|---|---|---|---|
| `pd_known_at` | datetime | `RUN_DATETIME` = 13:00 on D-1. Conservative: the run is published shortly after 12:30 | Leakage control |
| `pd_rrp` | float, $/MWh | AEMO's forecast price. `pd_rrp < 0` is AEMO's own negative-price forecast. Frozen at the $15,100 cap on several days in June 2022 | Benchmark; input |
| `pd_totaldemand` | float, MW | Forecast of `totaldemand` (AEMO demand forecast; uses forecast temperature and the rooftop PV forecast) | Input |
| `pd_availablegeneration` | float, MW | Forecast of `availablegeneration` (from bid availabilities) | Input |
| `pd_netinterchange` | float, MW | Forecast of `netinterchange` | Input |
| `pd_demand_and_nonschedgen` | float, MW | Forecast of `demand_and_nonschedgen` | Input |
| `pd_uigf` | float, MW | Forecast unconstrained output of semi-scheduled wind + solar | Input |
| `pd_ss_solar_uigf` | float, MW | Solar-farm part of `pd_uigf` | Input |
| `pd_ss_wind_uigf` | float, MW | Wind-farm part of `pd_uigf` | Input |

## AEMO rooftop PV forecast (`pvf_`)

Version used for day D: the latest labelled ≤ 12:30 on D-1, the same one AEMO's 13:00 predispatch would have used. Exception: AEMO forecast outage, so target days 15, 16 and 17 Dec 2024 use the 14 Dec 08:00 version (5, 29 and 53 h old at issue time).

| Column | Type | Description | Role |
|---|---|---|---|
| `pvf_known_at` | datetime | `VERSION_DATETIME`. In AEMO's database 8 min earlier (`LASTCHANGED`) | Leakage control |
| `pvf_powermean` | float, MW | Mean forecast of rooftop PV output | Input |
| `pvf_powerpoe50` | float, MW | Median forecast (50% probability of exceedance) | Input |
| `pvf_powerpoelow` | float, MW | Lower end of AEMO's forecast range | Input |
| `pvf_powerpoehigh` | float, MW | Upper end of AEMO's forecast range | Input |

`pvf_powerpoelow ≤ pvf_powerpoe50 ≤ pvf_powerpoehigh` in every row (checked). The exact probability levels of the low and high bounds are defined in AEMO's MMS data model; check them before using the bounds as prediction intervals.

## SILO weather (`silo_`)

Observed daily weather from SILO DataDrill (gridded) at Adelaide (SA1), Melbourne (VIC1) and Sydney (NSW1). Joined by `DAY`, so every half-hour of day D carries **day D's actual weather**. That is a leak for the main model. Main model: use values from D-2 (complete by 09:00 on D-1). What-if model: day D as is. SILO revises recent days as late station reports arrive, so a fresh download can differ slightly from what was available at the time.

| Column | Unit | Description | Role |
|---|---|---|---|
| `silo_daily_rain` | mm | Rainfall, 24 h to 09:00 on that date | D-2: input · D: what-if only |
| `silo_max_temp` | °C | Maximum temperature (Bureau of Meteorology 9 am convention) | D-2: input · D: what-if only |
| `silo_min_temp` | °C | Minimum temperature, 24 h to 09:00 | D-2: input · D: what-if only |
| `silo_radiation` | MJ/m² | Daily total incoming solar radiation | D-2: input · D: what-if only |
| `silo_vp` | hPa | Vapour pressure (humidity) | D-2: input · D: what-if only |
| `silo_known_at` | datetime | `DAY` + 1 day + 09:00: the earliest moment day D's values are complete. SILO's actual publication lag is unknown; the D-2 rule leaves margin | Leakage control |

## Flags

| Column | Type | Description | Role |
|---|---|---|---|
| `admin_price` | bool | True on target days with administered prices (June 2022 energy crisis: $300 price cap, then market suspension; AEMO's forecast frozen at $15,100). Window in `config.ADMIN_PRICE_DAYS`: 13–24 Jun 2022 (first $300 price NSW 13 Jun 19:30, last 23 Jun 01:30; AEMO forecast frozen at the cap for whole days only inside this window). 1,728 half-hours (0.66%). Kept for training; **excluded from backtest metrics and the battery dollar figure** | Scoring filter |

## Known gaps and special periods

| Period | Affects | Treatment |
|---|---|---|
| Target days 2 Oct – 1 Nov 2022 | All `pd_` columns missing | XGBoost handles missing values; benchmark comparison excludes these days |
| 13–24 Jun 2022 | Actual and forecast prices administered | `admin_price` flag |
| 15–17 Dec 2024 | `pvf_` from an old version (AEMO outage) | Kept: predispatch price and demand errors on those days were within the normal range |
| 34 half-hours | `pv_actual_mw` missing | Left missing |
