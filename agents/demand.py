"""Demand forecasting agent — standalone module, Snowflake Cortex ML.

Forecasts daily mission demand per base over a horizon using
SNOWFLAKE.ML.FORECAST: one multi-series model, series keyed by base.
Demand = civilian medical-transport and cargo request volume (mission
requests per base per day).

Returns point forecasts with prediction intervals (lower/upper). A base
without enough history is flagged "insufficient history" rather than
forecast — the model never gets handed a series it can't learn.

Note: the synthetic history ends 2025-12-30, so forecasts cover the days
immediately after that, not today-forward. A live system would retrain on
data through the current date.

Usage:
    python -m agents.demand
"""

from dataclasses import dataclass, field
from datetime import date

from utils.snowflake import get_connection

HORIZON_DAYS = 14
MIN_HISTORY_DAYS = 60          # below this we don't trust a forecast — flag it
PREDICTION_INTERVAL = 0.95
MODEL_NAME = "demand_model"
TRAIN_VIEW = "demand_train"


@dataclass(frozen=True)
class DemandPoint:
    date: date
    forecast: float
    lower: float            # prediction-interval bounds at PREDICTION_INTERVAL
    upper: float


@dataclass(frozen=True)
class BaseDemandForecast:
    base_id: int
    name: str
    horizon_days: int
    sufficient_history: bool
    history_days: int
    points: list[DemandPoint] = field(default_factory=list)  # empty if insufficient
    note: str | None = None


# -- data fetching -----------------------------------------------------------

def _get_bases(cur):
    cur.execute("SELECT base_id, name FROM bases ORDER BY base_id")
    return {r[0]: r[1] for r in cur.fetchall()}


def _series_history(cur):
    """Distinct demand-days per base — gates the cold-start decision."""
    cur.execute("""
        SELECT origin_base_id, COUNT(DISTINCT requested_date::DATE)
        FROM missions GROUP BY 1
    """)
    return {r[0]: int(r[1]) for r in cur.fetchall()}


# -- cold-start gating (pure, unit-testable) ---------------------------------

def _classify(base_ids, history):
    """Split bases into those with enough history to forecast and those flagged."""
    eligible, insufficient = [], []
    for b in base_ids:
        target = eligible if history.get(b, 0) >= MIN_HISTORY_DAYS else insufficient
        target.append(b)
    return eligible, insufficient


def _insufficient_result(base_id, name, history_days, horizon):
    return BaseDemandForecast(
        base_id=base_id, name=name, horizon_days=horizon,
        sufficient_history=False, history_days=history_days, points=[],
        note=f"insufficient history: {history_days}d < {MIN_HISTORY_DAYS}d minimum",
    )


# -- Cortex ML ---------------------------------------------------------------

def _train_and_forecast(cur, eligible, horizon):
    """Train one multi-series Cortex model and forecast the horizon.

    Returns raw (series, ts, forecast, lower, upper) rows.
    """
    ids = ",".join(str(b) for b in eligible)
    cur.execute(f"""
        CREATE OR REPLACE TEMPORARY VIEW {TRAIN_VIEW} AS
        SELECT origin_base_id AS base_id,
               DATE_TRUNC('day', requested_date) AS ds,
               COUNT(*)::FLOAT AS y
        FROM missions
        WHERE origin_base_id IN ({ids})
        GROUP BY 1, 2
    """)
    cur.execute(f"""
        CREATE OR REPLACE SNOWFLAKE.ML.FORECAST {MODEL_NAME}(
            INPUT_DATA => TABLE({TRAIN_VIEW}),
            SERIES_COLNAME => 'base_id',
            TIMESTAMP_COLNAME => 'ds',
            TARGET_COLNAME => 'y'
        )
    """)
    cur.execute(f"""
        CALL {MODEL_NAME}!FORECAST(
            FORECASTING_PERIODS => {horizon},
            CONFIG_OBJECT => {{'prediction_interval': {PREDICTION_INTERVAL}}}
        )
    """)
    cur.execute("""
        SELECT series, ts, forecast, lower_bound, upper_bound
        FROM TABLE(RESULT_SCAN(LAST_QUERY_ID()))
        ORDER BY series, ts
    """)
    return cur.fetchall()


def _build_forecasts(rows, bases, eligible, insufficient, horizon, history):
    """Map Cortex output rows to dataclasses; merge in flagged cold-start bases."""
    by_series = {}
    for series, ts, fc, lo, hi in rows:
        bid = int(str(series).strip('"'))
        by_series.setdefault(bid, []).append(DemandPoint(
            date=ts.date(),
            forecast=round(float(fc), 2),
            lower=round(float(lo), 2),
            upper=round(float(hi), 2),
        ))

    results = [
        BaseDemandForecast(
            base_id=b, name=bases.get(b, "?"), horizon_days=horizon,
            sufficient_history=True, history_days=history.get(b, 0),
            points=by_series.get(b, []),
        )
        for b in eligible
    ]
    results += [
        _insufficient_result(b, bases.get(b, "?"), history.get(b, 0), horizon)
        for b in insufficient
    ]
    results.sort(key=lambda r: r.base_id)
    return results


# -- public API --------------------------------------------------------------

def forecast(conn=None, horizon_days=HORIZON_DAYS) -> list[BaseDemandForecast]:
    """Forecast daily mission demand per base over the horizon."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    cur = conn.cursor()
    try:
        bases = _get_bases(cur)
        history = _series_history(cur)
        eligible, insufficient = _classify(list(bases), history)
        rows = _train_and_forecast(cur, eligible, horizon_days) if eligible else []
        return _build_forecasts(rows, bases, eligible, insufficient, horizon_days, history)
    finally:
        cur.close()
        if own_conn:
            conn.close()


def to_dict(results: list[BaseDemandForecast]) -> list[dict]:
    """JSON-serializable view — the orchestrator's input contract."""
    return [
        {
            "base_id": r.base_id, "name": r.name,
            "horizon_days": r.horizon_days,
            "sufficient_history": r.sufficient_history,
            "history_days": r.history_days,
            "note": r.note,
            "points": [
                {"date": p.date.isoformat(), "forecast": p.forecast,
                 "lower": p.lower, "upper": p.upper}
                for p in r.points
            ],
        }
        for r in results
    ]


# -- CLI output --------------------------------------------------------------

def print_report(results: list[BaseDemandForecast]):
    horizon = results[0].horizon_days if results else 0
    print(f"\n{'='*70}")
    print(f"DEMAND FORECAST - next {horizon}d, {len(results)} bases")
    print(f"{'='*70}")
    print(f"  {'Base':<28} {'Hist':>5} {'Avg/day':>8}  {'Interval (avg)':<18}")
    print(f"  {'-'*64}")
    for r in sorted(results, key=lambda x: x.base_id):
        if not r.sufficient_history:
            print(f"  {r.name:<28} {r.history_days:>5}  INSUFFICIENT HISTORY")
            continue
        n = len(r.points) or 1
        avg = sum(p.forecast for p in r.points) / n
        lo = sum(p.lower for p in r.points) / n
        hi = sum(p.upper for p in r.points) / n
        print(f"  {r.name:<28} {r.history_days:>5} {avg:>8.1f}  [{lo:.1f}, {hi:.1f}]")


if __name__ == "__main__":
    print_report(forecast())
