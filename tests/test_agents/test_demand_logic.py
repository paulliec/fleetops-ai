"""Demand agent logic tests — cold-start gating and Cortex-row mapping.

Hermetic: the pure functions are tested without Snowflake or Cortex (the
live training run is exercised separately via `py -m agents.demand`).
Run with pytest or directly:

    py -m tests.test_agents.test_demand_logic
"""

from datetime import datetime

from agents import demand
from agents.demand import MIN_HISTORY_DAYS


def test_classify_splits_on_history():
    history = {1: 730, 2: MIN_HISTORY_DAYS, 3: MIN_HISTORY_DAYS - 1, 4: 0}
    eligible, insufficient = demand._classify([1, 2, 3, 4], history)
    assert eligible == [1, 2]           # >= threshold
    assert insufficient == [3, 4]       # below, and a base with no history at all


def test_insufficient_result_is_clean():
    r = demand._insufficient_result(7, "Test Base", 10, 14)
    assert r.sufficient_history is False
    assert r.points == []
    assert r.history_days == 10
    assert "insufficient history" in r.note
    assert "10d" in r.note and f"{MIN_HISTORY_DAYS}d" in r.note


def test_build_forecasts_maps_intervals_and_merges():
    bases = {1: "Alpha", 2: "Bravo"}
    history = {1: 730, 2: 5}
    # base 1 eligible (two Cortex rows), base 2 flagged cold-start
    rows = [
        (1, datetime(2026, 1, 1), 8.4, 6.1, 10.7),
        (1, datetime(2026, 1, 2), 7.9, 5.6, 10.2),
    ]
    results = demand._build_forecasts(rows, bases, [1], [2], 14, history)

    assert [r.base_id for r in results] == [1, 2]          # sorted, both present
    a, b = results
    assert a.sufficient_history and len(a.points) == 2
    p = a.points[0]
    assert (p.forecast, p.lower, p.upper) == (8.4, 6.1, 10.7)
    assert p.lower < p.forecast < p.upper                  # interval brackets the point
    assert b.sufficient_history is False and b.points == []


def test_to_dict_serializes():
    bases = {1: "Alpha"}
    rows = [(1, datetime(2026, 1, 1), 8.4, 6.1, 10.7)]
    results = demand._build_forecasts(rows, bases, [1], [], 14, {1: 730})
    d = demand.to_dict(results)
    assert d[0]["points"][0]["date"] == "2026-01-01"
    assert d[0]["sufficient_history"] is True


if __name__ == "__main__":
    test_classify_splits_on_history()
    print("PASS: cold-start gating splits on history threshold")
    test_insufficient_result_is_clean()
    print("PASS: insufficient-history path returns clean flagged result")
    test_build_forecasts_maps_intervals_and_merges()
    print("PASS: Cortex rows mapped with intervals, cold-start bases merged")
    test_to_dict_serializes()
    print("PASS: to_dict produces serializable orchestrator contract")
