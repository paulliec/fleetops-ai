"""Staffing agent logic tests — eligibility filters and greedy allocation.

Hermetic: the pure functions are tested with plain data, no Snowflake (the
live coverage run is exercised via `py -m agents.staffing`). Run with pytest
or directly:

    py -m tests.test_agents.test_staffing_logic
"""

from agents import staffing
from agents.staffing import AircraftSlot, build_pool, build_base_coverage


def _crew(crew_id, role, quals, status="available"):
    return {"crew_id": crew_id, "role": role, "status": status, "quals": set(quals)}


# a full EC135 complement: pilot + flight_nurse + flight_paramedic
def _ec135_pool():
    return build_pool([
        _crew(1, "pilot", ["EC135"]),
        _crew(2, "flight_nurse", ["EC135"]),
        _crew(3, "flight_paramedic", ["EC135"]),
    ], resting_pilot_ids=set())


def test_status_filter_excludes_unavailable():
    rows = [
        _crew(1, "pilot", ["EC135"], status="available"),
        _crew(2, "pilot", ["EC135"], status="on_leave"),
        _crew(3, "pilot", ["EC135"], status="medical"),
        _crew(4, "pilot", ["EC135"], status="on_assignment"),
    ]
    pool = build_pool(rows, resting_pilot_ids=set())
    assert [c.crew_id for c in pool] == [1]


def test_rest_filter_excludes_recent_pilot():
    rows = [
        _crew(1, "pilot", ["EC135"]),            # resting -> excluded
        _crew(2, "flight_nurse", ["EC135"]),     # same id space, not a pilot -> kept
    ]
    pool = build_pool(rows, resting_pilot_ids={1})
    assert {c.crew_id for c in pool} == {2}

    # a nurse who happens to share id 1 is NOT gated by the pilot rest set
    nurse_only = build_pool([_crew(1, "flight_nurse", ["EC135"])], resting_pilot_ids={1})
    assert {c.crew_id for c in nurse_only} == {1}


def test_qualification_excludes_nonqualified():
    # pilot qualified only on Bell 407 can't crew the EC135 seat
    pool = build_pool([
        _crew(1, "pilot", ["Bell 407"]),
        _crew(2, "flight_nurse", ["EC135"]),
        _crew(3, "flight_paramedic", ["EC135"]),
    ], resting_pilot_ids=set())
    slots = [AircraftSlot(10, "N10", "EC135")]
    cov = build_base_coverage(1, "Test", slots, pool)
    ac = cov.aircraft[0]
    assert ac.crewed is False
    assert [(s.role, s.short) for s in ac.shortfalls] == [("pilot", 1)]


def test_specific_shortfall():
    # full pool minus the paramedic -> short exactly one flight_paramedic
    pool = build_pool([
        _crew(1, "pilot", ["EC135"]),
        _crew(2, "flight_nurse", ["EC135"]),
    ], resting_pilot_ids=set())
    slots = [AircraftSlot(10, "N10", "EC135")]
    cov = build_base_coverage(1, "Test", slots, pool)
    ac = cov.aircraft[0]
    assert ac.crewed is False
    assert len(ac.shortfalls) == 1
    assert ac.shortfalls[0].role == "flight_paramedic"
    assert ac.shortfalls[0].short == 1
    assert cov.status == "critical"


def test_shared_crew_no_overcount():
    """Two EC135 but only one pilot qualified on EC135 -> exactly one crewed,
    the other short a pilot. Guards against per-type independent counting."""
    pool = build_pool([
        _crew(1, "pilot", ["EC135"]),
        _crew(2, "flight_nurse", ["EC135"]),
        _crew(3, "flight_paramedic", ["EC135"]),
        _crew(4, "flight_nurse", ["EC135"]),
        _crew(5, "flight_paramedic", ["EC135"]),
    ], resting_pilot_ids=set())
    slots = [AircraftSlot(10, "N10", "EC135"), AircraftSlot(11, "N11", "EC135")]
    cov = build_base_coverage(1, "Test", slots, pool)
    assert cov.crewed_count == 1
    assert cov.status == "partial"
    uncrewed = [a for a in cov.aircraft if not a.crewed]
    assert len(uncrewed) == 1
    assert [(s.role, s.short) for s in uncrewed[0].shortfalls] == [("pilot", 1)]


def test_coverage_status():
    assert staffing.coverage_status(2, 2) == "full"
    assert staffing.coverage_status(1, 2) == "partial"
    assert staffing.coverage_status(0, 2) == "critical"
    assert staffing.coverage_status(0, 0) == "full"   # nothing to crew


if __name__ == "__main__":
    test_status_filter_excludes_unavailable()
    print("PASS: status filter excludes unavailable crew")
    test_rest_filter_excludes_recent_pilot()
    print("PASS: rest filter excludes recent pilots, not other roles")
    test_qualification_excludes_nonqualified()
    print("PASS: qualification filter excludes non-qualified crew")
    test_specific_shortfall()
    print("PASS: missing paramedic surfaces specific 'short 1 flight_paramedic'")
    test_shared_crew_no_overcount()
    print("PASS: shared crew not double-counted across same-type aircraft")
    test_coverage_status()
    print("PASS: coverage status thresholds (full/partial/critical)")
