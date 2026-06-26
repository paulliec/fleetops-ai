"""Orchestrator synthesis tests — safety gate, grounding, decay, conflict.

Hermetic: builds region bundles by hand and injects a fake Cortex COMPLETE,
so no Snowflake and no LLM. The live end-to-end run is exercised via
`py -m graph.workflow`. Run with pytest or directly:

    py -m tests.test_graph.test_orchestrator
"""

import json
import re

from graph import orchestrator as orch
from graph.orchestrator import LEVERS, RegionBundle


# -- fixtures: agent-shaped dicts (match each agent's to_dict output) ---------

def _weather(base_id=1, name="Test Base", no_fly=False):
    if no_fly:
        windows = [{"category": "no_fly", "start": "2026-06-26T10:00:00",
                    "end": "2026-06-26T13:00:00", "hours": 3, "factors": ["weather"]}]
    else:
        windows = [{"category": "flyable", "start": "2026-06-26T10:00:00",
                    "end": "2026-06-27T10:00:00", "hours": 24, "factors": []}]
    return {"base_id": base_id, "name": name, "icao_code": "KTST", "lat": 33.0, "lon": -84.0,
            "flyable_hours": 0 if no_fly else 24, "marginal_hours": 0,
            "no_fly_hours": 3 if no_fly else 0,
            "next_no_fly": windows[0] if no_fly else None, "windows": windows}


def _demand(base_id=1, name="Test Base", avg=8.0, width=4.0, sufficient=True):
    pts = [{"date": "2026-06-26", "forecast": avg,
            "lower": avg - width / 2, "upper": avg + width / 2}]
    return {"base_id": base_id, "name": name, "horizon_days": 14,
            "sufficient_history": sufficient, "history_days": 730, "note": None,
            "points": pts if sufficient else []}


def _staffing(base_id=1, name="Test Base", active=3, crewed=1,
              short_role="flight_nurse", short_n=1, avail=None):
    avail = {"pilot": 2, "copilot": 1} if avail is None else avail
    aircraft = [{"aircraft_id": 10, "tail_number": "N10", "aircraft_type": "EC135",
                 "crewed": False, "assigned_crew_ids": [],
                 "shortfalls": [{"role": short_role, "short": short_n}]}]
    return {"base_id": base_id, "name": name, "active_aircraft": active,
            "crewed_count": crewed, "status": "partial",
            "available_by_role": avail, "aircraft": aircraft}


def _maint(aircraft_id=10, tail="N10", hard=False, days=5):
    nd = {"category": "airframe_life" if hard else "engine",
          "trigger": "life_limit" if hard else "hours", "due_date": None,
          "hours_remaining": 0 if hard else 50,
          "days_remaining": 0 if hard else days, "hard_ground": hard}
    return {"aircraft_id": aircraft_id, "tail_number": tail, "aircraft_type": "EC135",
            "total_hours": 5000, "rate_30d": 2.0, "rate_90d": 2.0,
            "tempo_flag": False, "next_due": nd, "consolidation": []}


def _bundle(weather=None, demand=None, staffing=None, maint=None):
    w = [_weather()] if weather is None else weather
    d = [_demand()] if demand is None else demand
    s = [_staffing()] if staffing is None else staffing
    m = [] if maint is None else maint
    bases = [{"base_id": 1, "name": "Test Base", "region": "TestRegion",
              "lat": 33.0, "lon": -84.0}]
    return RegionBundle("TestRegion", bases, m, w, d, s, orch._safety(w, m, s))


def _gctx(bundle):
    return orch.build_global_ctx([bundle])


def fake_complete(prompt):
    """Parse candidate ids out of the prompt and rank them in REVERSE, so the
    test can prove the model's ordering is actually applied."""
    ids = [int(x) for x in re.findall(r"\[(\d+)\] action=", prompt)]
    items = [{"candidate_id": cid, "priority_rank": i + 1, "horizon_days": 2,
              "pillars_addressed": ["staffing"], "rationale": f"cite candidate {cid}"}
             for i, cid in enumerate(reversed(ids))]
    return json.dumps(items)


# -- confidence / geometry ---------------------------------------------------

def test_horizon_decay_by_pillar():
    # weather decays fast; maintenance holds. Same 10-day horizon, opposite confidence.
    assert orch.horizon_confidence("weather", 10) < 0.05
    assert orch.horizon_confidence("maintenance", 10) > 0.8
    # monotonic decay with horizon
    assert orch.horizon_confidence("demand", 1) > orch.horizon_confidence("demand", 20)


def test_demand_confidence_widens_to_lower():
    narrow = orch.demand_confidence(3, avg=8.0, interval_width=2.0)
    wide = orch.demand_confidence(3, avg=8.0, interval_width=12.0)
    assert wide < narrow            # wider interval -> lower confidence


def test_great_circle():
    a = {"lat": 33.0, "lon": -84.0}
    assert orch.great_circle_nm(a, a) == 0.0
    assert orch.great_circle_nm(a, {"lat": 29.65, "lon": -95.28}) > 400


# -- safety gate (hard, before the LLM) --------------------------------------

def test_gate_removes_flight_option_under_no_fly():
    bundle = _bundle(weather=[_weather(no_fly=True)], demand=[_demand(avg=9.0)])
    cands = orch.generate_candidates(bundle, _gctx(bundle))
    survivors, blocked = orch.safety_gate(cands, bundle)
    # prestage (requires flight) must be blocked, never reach ranking
    assert any(c.action == "prestage_for_demand" for c in blocked)
    assert all(c.action != "prestage_for_demand" for c in survivors)
    assert all(c.gate_reason == "active no-fly window"
               for c in blocked if c.action == "prestage_for_demand")


def test_gate_removes_poolfirst_crew_under_rest_gap():
    # short flight_nurse, none available -> rest gap -> pool-first crew move blocked
    bundle = _bundle(staffing=[_staffing(short_role="flight_nurse",
                                          avail={"pilot": 2})])
    assert "flight_nurse" in bundle.safety["rest_gaps"]
    cands = orch.generate_candidates(bundle, _gctx(bundle))
    _survivors, blocked = orch.safety_gate(cands, bundle)
    pool_blocked = [c for c in blocked
                    if c.action == "reposition_crew" and c.extra.get("mode") == "pool"]
    assert pool_blocked
    assert pool_blocked[0].gate_reason == "no rested crew available in pool"


# -- candidate generation / conflict -----------------------------------------

def test_surge_with_staffing_gap_surfaces_both_pillars():
    """Demand says surge, staffing can't crew it — the conflict must show in
    the candidate, not be papered over."""
    bundle = _bundle(demand=[_demand(avg=9.0)],
                     staffing=[_staffing(active=3, crewed=1)])
    cands = orch.generate_candidates(bundle, _gctx(bundle))
    prestage = next(c for c in cands if c.action == "prestage_for_demand")
    assert "demand" in prestage.pillars and "staffing" in prestage.pillars
    assert "demand" in prestage.supporting_signals
    assert "staffing" in prestage.supporting_signals


def test_actions_are_constrained_to_lever_set():
    bundle = _bundle(maint=[_maint(hard=True)], demand=[_demand(avg=9.0)])
    cands = orch.generate_candidates(bundle, _gctx(bundle))
    assert cands and all(c.action in LEVERS for c in cands)


# -- end-to-end region planning (mocked LLM) ---------------------------------

def test_plan_region_grounded_and_ranked():
    bundle = _bundle(demand=[_demand(avg=9.0)],
                     staffing=[_staffing(active=3, crewed=1)],
                     maint=[_maint(hard=True)])
    packages, _blocked = orch.plan_region(bundle, _gctx(bundle), fake_complete)

    assert packages
    # every action is a real lever
    assert all(p.action in LEVERS for p in packages)
    # grounding: signals are code-sourced, keys only from real pillars
    allowed = {"maintenance", "weather", "demand", "staffing", "logistics"}
    for p in packages:
        assert p.supporting_signals
        assert set(p.supporting_signals).issubset(allowed)
    # priority ranks are dense 1..N after renumber
    assert sorted(p.priority_rank for p in packages) == list(range(1, len(packages) + 1))


def test_plan_region_applies_model_ordering():
    bundle = _bundle(demand=[_demand(avg=9.0)], staffing=[_staffing()])
    survivors, _ = orch.safety_gate(orch.generate_candidates(bundle, _gctx(bundle)), bundle)
    packages, _blocked = orch.plan_region(bundle, _gctx(bundle), fake_complete)
    # fake ranks reversed; first package should map to the LAST survivor cid
    last_cid_action = max(survivors, key=lambda c: c.cid).action
    assert packages[0].action == last_cid_action


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS: {name}")
