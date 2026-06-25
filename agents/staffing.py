"""Staffing coverage agent — standalone module, deterministic (no LLM, no ML).

Reports the SUPPLY side of crew: for each base, how many of its active
aircraft can be fully crewed right now against a fixed minimum-complement
policy, plus the specific gaps where they can't. It does NOT look at
demand, weather, or maintenance — reconciling supply against demand is the
orchestrator's job. Coverage here is measured against a staffing standard,
not workload.

Crew complement, qualification, availability, and rest are rules with a
correct answer, so this is deterministic by design — the inverse of the
demand forecaster.

Simplifications (real ops are more complex):
- Duty/rest is a simplified FAA Part 135-style rule: a pilot who landed
  within MIN_REST_HOURS isn't yet rested. Real rules track cumulative duty
  windows, lookback periods, and rest opportunities.
- Rest is only derivable for PIC pilots (the only crew recorded in
  flight_logs). Copilots/medical crew have no duty record here, so they're
  gated by status only.
- Qualification = membership in the crew's `qualifications` array. Real
  currency tracks recent landings per type, which this data doesn't carry.
- One complement per aircraft (mission-ready once), not shift-aware.
- Allocation is greedy, not an optimal matching — it never over-reports
  coverage, but can under-report in rare high-overlap cases (see allocate_base).

Usage:
    python -m agents.staffing
"""

import json
from dataclasses import dataclass, field

from utils.snowflake import get_connection

MIN_REST_HOURS = 10

# minimum crew complement per aircraft type — staffing policy, not real ops minima
COMPLEMENTS = {
    "Bell 407":     {"pilot": 1, "flight_nurse": 1, "flight_paramedic": 1},
    "EC135":        {"pilot": 1, "flight_nurse": 1, "flight_paramedic": 1},
    "AW139":        {"pilot": 1, "copilot": 1, "flight_nurse": 1, "flight_paramedic": 1},
    "King Air 350": {"pilot": 1, "copilot": 1},
    "PC-12":        {"pilot": 1},
    "Citation CJ3": {"pilot": 1, "copilot": 1},
}

# lower = scarcer pool = allocate first. paramedic/nurse qualify on fewest types.
ROLE_SCARCITY = {"flight_paramedic": 0, "flight_nurse": 1, "copilot": 2, "pilot": 3}


@dataclass(frozen=True)
class CrewMember:
    crew_id: int
    role: str
    quals: frozenset


@dataclass(frozen=True)
class AircraftSlot:
    aircraft_id: int
    tail_number: str
    aircraft_type: str


@dataclass(frozen=True)
class Shortfall:
    role: str
    short: int


@dataclass(frozen=True)
class AircraftCoverage:
    aircraft_id: int
    tail_number: str
    aircraft_type: str
    crewed: bool
    assigned_crew_ids: list[int] = field(default_factory=list)
    shortfalls: list[Shortfall] = field(default_factory=list)


@dataclass(frozen=True)
class BaseCoverage:
    base_id: int
    name: str
    active_aircraft: int
    crewed_count: int
    status: str                 # full | partial | critical
    available_by_role: dict = field(default_factory=dict)
    aircraft: list[AircraftCoverage] = field(default_factory=list)


# -- coverage logic (pure, no Snowflake) -------------------------------------

def build_pool(crew_rows, resting_pilot_ids):
    """Eligible crew at a base: available status, and rested (pilots only —
    other roles have no duty record so status is the only gate)."""
    pool = []
    for c in crew_rows:
        if c["status"] != "available":
            continue
        if c["role"] == "pilot" and c["crew_id"] in resting_pilot_ids:
            continue
        pool.append(CrewMember(c["crew_id"], c["role"], frozenset(c["quals"])))
    return pool


def aircraft_sort_key(slot):
    """Most-constrained aircraft first: largest complement, then scarcest
    required role, then id. Crew them while the pool is full."""
    comp = COMPLEMENTS.get(slot.aircraft_type, {})
    size = sum(comp.values())
    scarcest = min((ROLE_SCARCITY[r] for r in comp), default=99)
    return (-size, scarcest, slot.aircraft_id)


def allocate_base(slots, pool):
    """Greedy per-base crew allocation.

    For each aircraft (most-constrained first), fill each seat from the
    not-yet-assigned pool, preferring least-qualified crew so versatile crew
    stay free for seats only they can fill. Commit only on a full complement;
    a short aircraft releases its tentative picks for others to use.

    Conservative: never over-reports coverage (committed crew leave the pool).
    Can under-report vs an optimal bipartite matching in rare high-overlap
    cases — acceptable, and it yields the per-aircraft shortfall for free.
    """
    assigned = set()
    coverages = []
    for slot in sorted(slots, key=aircraft_sort_key):
        comp = COMPLEMENTS.get(slot.aircraft_type, {})
        tentative, shortfalls = [], []
        for role, need in comp.items():
            candidates = [
                c for c in pool
                if c.crew_id not in assigned
                and c.role == role
                and slot.aircraft_type in c.quals
            ]
            candidates.sort(key=lambda c: (len(c.quals), c.crew_id))
            pick = candidates[:need]
            if len(pick) < need:
                shortfalls.append(Shortfall(role, need - len(pick)))
            tentative.extend(pick)

        if shortfalls:
            coverages.append(AircraftCoverage(
                slot.aircraft_id, slot.tail_number, slot.aircraft_type,
                crewed=False, shortfalls=shortfalls))
        else:
            assigned.update(c.crew_id for c in tentative)
            coverages.append(AircraftCoverage(
                slot.aircraft_id, slot.tail_number, slot.aircraft_type,
                crewed=True, assigned_crew_ids=[c.crew_id for c in tentative]))

    available_by_role = {}
    for c in pool:
        available_by_role[c.role] = available_by_role.get(c.role, 0) + 1

    coverages.sort(key=lambda a: a.aircraft_id)
    return coverages, available_by_role


def coverage_status(crewed_count, active_aircraft):
    if active_aircraft == 0 or crewed_count == active_aircraft:
        return "full"
    if crewed_count == 0:
        return "critical"
    return "partial"


def build_base_coverage(base_id, name, slots, pool):
    coverages, available_by_role = allocate_base(slots, pool)
    crewed = sum(1 for c in coverages if c.crewed)
    return BaseCoverage(
        base_id=base_id, name=name, active_aircraft=len(slots),
        crewed_count=crewed, status=coverage_status(crewed, len(slots)),
        available_by_role=available_by_role, aircraft=coverages,
    )


# -- data fetching -----------------------------------------------------------

def _get_as_of(cur):
    cur.execute("SELECT MAX(scheduled_departure) FROM flight_logs")
    return cur.fetchone()[0]


def _get_bases(cur):
    cur.execute("SELECT base_id, name FROM bases ORDER BY base_id")
    return {r[0]: r[1] for r in cur.fetchall()}


def _parse_quals(raw):
    if raw is None:
        return set()
    if isinstance(raw, (list, tuple)):
        return set(raw)
    return set(json.loads(raw))     # Snowflake ARRAY comes back as a JSON string


def _get_crew(cur):
    cur.execute("""
        SELECT crew_id, role, base_id, status, qualifications
        FROM crew ORDER BY crew_id
    """)
    return [
        {"crew_id": cid, "role": role, "base_id": bid,
         "status": status, "quals": _parse_quals(quals)}
        for cid, role, bid, status, quals in cur.fetchall()
    ]


def _get_active_aircraft(cur):
    cur.execute("""
        SELECT aircraft_id, tail_number, aircraft_type, home_base_id
        FROM aircraft WHERE status = 'active' ORDER BY aircraft_id
    """)
    by_base = {}
    for aid, tail, atype, hbase in cur.fetchall():
        by_base.setdefault(hbase, []).append(AircraftSlot(aid, tail, atype))
    return by_base


def _get_resting_pilots(cur, as_of):
    """Pilots who landed within MIN_REST_HOURS of as_of — still resting.

    Set contains only violators, so everyone else is eligible by omission:
    no flights / old flight / null arrival all fall out correctly.
    """
    cur.execute("""
        SELECT pic_crew_id FROM flight_logs
        WHERE actual_arrival IS NOT NULL
        GROUP BY pic_crew_id
        HAVING MAX(actual_arrival) > DATEADD(hour, %(h)s, %(d)s)
    """, {"h": -MIN_REST_HOURS, "d": as_of})
    return {r[0] for r in cur.fetchall()}


# -- public API --------------------------------------------------------------

def assess(conn=None) -> list[BaseCoverage]:
    """Assess crew coverage for all bases."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    cur = conn.cursor()
    try:
        as_of = _get_as_of(cur)
        bases = _get_bases(cur)
        crew = _get_crew(cur)
        aircraft_by_base = _get_active_aircraft(cur)
        resting = _get_resting_pilots(cur, as_of)
    finally:
        cur.close()
        if own_conn:
            conn.close()

    results = []
    for base_id, name in bases.items():
        base_crew = [c for c in crew if c["base_id"] == base_id]
        pool = build_pool(base_crew, resting)
        slots = aircraft_by_base.get(base_id, [])
        results.append(build_base_coverage(base_id, name, slots, pool))
    results.sort(key=lambda r: r.base_id)
    return results


def to_dict(results: list[BaseCoverage]) -> list[dict]:
    """JSON-serializable view — the orchestrator's input contract."""
    return [
        {
            "base_id": r.base_id, "name": r.name,
            "active_aircraft": r.active_aircraft,
            "crewed_count": r.crewed_count,
            "status": r.status,
            "available_by_role": r.available_by_role,
            "aircraft": [
                {
                    "aircraft_id": a.aircraft_id, "tail_number": a.tail_number,
                    "aircraft_type": a.aircraft_type, "crewed": a.crewed,
                    "assigned_crew_ids": a.assigned_crew_ids,
                    "shortfalls": [{"role": s.role, "short": s.short} for s in a.shortfalls],
                }
                for a in r.aircraft
            ],
        }
        for r in results
    ]


# -- CLI output --------------------------------------------------------------

def print_report(results: list[BaseCoverage]):
    print(f"\n{'='*70}")
    print(f"CREW COVERAGE - {len(results)} bases")
    print(f"{'='*70}")
    print(f"  {'Base':<28} {'Aircraft':>8} {'Crewed':>7} {'Status':>9}")
    print(f"  {'-'*56}")
    for r in sorted(results, key=lambda x: x.base_id):
        print(f"  {r.name:<28} {r.active_aircraft:>8} {r.crewed_count:>7} {r.status:>9}")

    for r in sorted(results, key=lambda x: x.base_id):
        gaps = [a for a in r.aircraft if not a.crewed]
        if not gaps:
            continue
        print(f"\n  {r.name} - gaps:")
        for a in gaps:
            short = ", ".join(f"short {s.short} {s.role}" for s in a.shortfalls)
            print(f"    {a.tail_number:<10} {a.aircraft_type:<14} {short}")


if __name__ == "__main__":
    print_report(assess())
