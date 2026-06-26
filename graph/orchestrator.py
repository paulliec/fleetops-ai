"""Orchestrator synthesis — turn the four agents' outputs into ranked,
conflict-aware, grounded decision packages, one synthesis per region.

Pipeline per region:
    1. assemble grounded signals from the four agents (their to_dict output)
    2. generate candidate actions from a CONSTRAINED lever set (deterministic)
    3. SAFETY GATE (deterministic, in code) — drop infeasible/unsafe options
       BEFORE the LLM ever sees them. Safety is a hard gate, never a ranking
       trade-off.
    4. rank the survivors + write cited rationale via Cortex COMPLETE (LLM)
    5. attach code-authoritative confidence (pillar-stability x horizon decay),
       logistics cost, and the grounded supporting signals
    6. persist to Snowflake (audit log + future feedback store)

Division of labor (this is the auditability payoff): code owns feasibility,
grounding, and the confidence math; the LLM only orders the already-safe,
already-grounded candidates and explains the trade-offs. It cannot invent an
action (constrained lever set) or a fact (signals are code-sourced).

Model: Cortex COMPLETE. Claude was NOT available via COMPLETE in this account
at build time (probed), so we fall back to the strongest available model and
leave a one-line seam to swap back. ML forecasting (demand) uses Cortex ML
FORECAST — same warehouse, different model family, each chosen for its job.

Logistics cost for crew/aircraft moves is great-circle nautical miles between
base lat/longs — a proxy. Real positioning cost (routing, fuel, crew travel)
is a later input.

Out of scope here (clean seams left, not built): Streamlit/Sigma UI, the
executive roll-up across regions, and any learning over the captured feedback.
"""

import json
import math
from dataclasses import dataclass, field, replace

from utils.snowflake import get_connection
from agents import maintenance, weather, demand, staffing

# Claude unavailable via Cortex COMPLETE in this account (probed) — strongest
# available fallback. Swap to a claude-* id here once enabled in the region.
CORTEX_MODEL = "llama3.1-70b"

DECISION_TABLE = "decision_packages"

LEVERS = (
    "reposition_aircraft", "reposition_crew", "schedule_maintenance",
    "prestage_for_demand", "flag_unresolved_gap",
)

# horizon/decay weighting: pillars differ in forecast stability, so confidence
# must decay differently with how far out the action applies.
PILLAR_STABILITY = {"maintenance": 1.0, "demand": 0.7, "staffing": 0.7, "weather": 0.4}
PILLAR_HALFLIFE_DAYS = {"maintenance": 60, "demand": 14, "staffing": 14, "weather": 1.5}

DUE_SOON_DAYS = 14        # maintenance due within this -> actionable now
SURGE_PER_DAY = 6.0       # demand avg/day at/above this -> prestage candidate

EARTH_RADIUS_NM = 3440.065


@dataclass(frozen=True)
class DecisionPackage:
    region: str
    action: str                       # from LEVERS
    pillars_addressed: list
    priority_rank: int
    horizon_days: int
    confidence: float
    supporting_signals: dict          # code-sourced; traces to agent output
    rationale: str                    # LLM; references the signals
    logistics_cost_nm: float | None = None
    feedback: dict | None = None      # {rating, note}; null until a user fills it


@dataclass
class Candidate:
    cid: int
    action: str
    primary_pillar: str
    pillars: list
    supporting_signals: dict
    requires: frozenset               # {"flight", "local_rested_crew"}
    default_horizon_days: int
    logistics_cost_nm: float | None = None
    extra: dict = field(default_factory=dict)
    gate_reason: str | None = None


@dataclass
class RegionBundle:
    region: str
    bases: list                       # [{base_id, name, region, lat, lon}]
    maintenance: list
    weather: list
    demand: list
    staffing: list
    safety: dict                      # {no_fly_now, groundings, rest_gaps}


@dataclass
class GlobalCtx:
    base_coords: dict                 # region -> {lat, lon, name}
    role_avail: dict                  # region -> {role: count}
    capacity: dict                    # region -> {crewed, active, status}

    def role_donor(self, role, exclude):
        for region, av in sorted(self.role_avail.items()):
            if region != exclude and av.get(role, 0) > 0:
                return {"region": region, "base": self.base_coords[region]}
        return None

    def capacity_donor(self, exclude):
        for region, cap in sorted(self.capacity.items()):
            if region != exclude and cap["status"] == "full" and cap["crewed"] >= 2:
                return {"region": region, "base": self.base_coords[region]}
        return None


# -- confidence (horizon x pillar-stability decay) ---------------------------

def horizon_confidence(pillar, horizon_days):
    base = PILLAR_STABILITY.get(pillar, 0.6)
    half_life = PILLAR_HALFLIFE_DAYS.get(pillar, 14)
    decay = 0.5 ** (max(horizon_days, 0) / half_life)
    return round(base * decay, 3)


def demand_confidence(horizon_days, avg, interval_width):
    """Demand confidence also shrinks with prediction-interval width — a wide
    band is itself low confidence, regardless of horizon."""
    c = horizon_confidence("demand", horizon_days)
    if avg and avg > 0:
        rel = min(interval_width / (avg * 2), 1.0)
        c *= (1 - 0.5 * rel)
    return round(c, 3)


def great_circle_nm(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, [a["lat"], a["lon"], b["lat"], b["lon"]])
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return round(2 * EARTH_RADIUS_NM * math.asin(math.sqrt(h)), 1)


# -- bundle assembly ---------------------------------------------------------

def _short_roles(st):
    out = {}
    for a in st["aircraft"]:
        for sf in a["shortfalls"]:
            out[sf["role"]] = out.get(sf["role"], 0) + sf["short"]
    return out


def _safety(weather_list, maint_list, staffing_list):
    no_fly_now = any(
        w["windows"] and w["windows"][0]["category"] == "no_fly"
        for w in weather_list
    )
    groundings = [m["tail_number"] for m in maint_list if m["next_due"]["hard_ground"]]
    rest_gaps = set()
    for st in staffing_list:
        for role in _short_roles(st):
            if st["available_by_role"].get(role, 0) == 0:
                rest_gaps.add(role)
    return {"no_fly_now": no_fly_now, "groundings": groundings, "rest_gaps": sorted(rest_gaps)}


def build_bundles(state, bases, aircraft_base):
    """Group the four agents' outputs by region. bases: base_id ->
    {name, region, lat, lon}. aircraft_base: aircraft_id -> base_id."""
    maint = maintenance.to_dict(state["maintenance"]) if state.get("maintenance") else []
    weath = weather.to_dict(state["weather"]) if state.get("weather") else []
    dem = demand.to_dict(state["demand"]) if state.get("demand") else []
    staff = staffing.to_dict(state["staffing"]) if state.get("staffing") else []

    regions = {}
    for bid, meta in bases.items():
        regions.setdefault(meta["region"], []).append({"base_id": bid, **meta})

    bundles = []
    for region, rbases in sorted(regions.items()):
        bids = {b["base_id"] for b in rbases}
        w = [x for x in weath if x["base_id"] in bids]
        d = [x for x in dem if x["base_id"] in bids]
        s = [x for x in staff if x["base_id"] in bids]
        m = [x for x in maint if aircraft_base.get(x["aircraft_id"]) in bids]
        bundles.append(RegionBundle(region, rbases, m, w, d, s, _safety(w, m, s)))
    return bundles


def build_global_ctx(bundles):
    base_coords, role_avail, capacity = {}, {}, {}
    for b in bundles:
        rb = b.bases[0]
        base_coords[b.region] = {"lat": rb["lat"], "lon": rb["lon"], "name": rb["name"]}
        ra = {}
        for st in b.staffing:
            for role, n in st["available_by_role"].items():
                ra[role] = ra.get(role, 0) + n
        role_avail[b.region] = ra
        crewed = sum(st["crewed_count"] for st in b.staffing)
        active = sum(st["active_aircraft"] for st in b.staffing)
        status = "full" if b.staffing and all(st["status"] == "full" for st in b.staffing) else "partial"
        capacity[b.region] = {"crewed": crewed, "active": active, "status": status}
    return GlobalCtx(base_coords, role_avail, capacity)


# -- candidate generation (deterministic, constrained lever set) -------------

def _staffing_for_base(bundle, base_id):
    return next((st for st in bundle.staffing if st["base_id"] == base_id), None)


def generate_candidates(bundle, gctx):
    cands = []
    counter = [0]

    def nxt():
        counter[0] += 1
        return counter[0]

    rbase = bundle.bases[0] if bundle.bases else None

    # maintenance: grounded or due soon -> schedule into a window
    for m in bundle.maintenance:
        nd = m["next_due"]
        days = nd["days_remaining"]
        if nd["hard_ground"] or (days is not None and days <= DUE_SOON_DAYS):
            horizon = 0 if nd["hard_ground"] else max(days, 0)
            tag = "hard ground" if nd["hard_ground"] else f"due {days}d"
            pillars = ["maintenance"]
            sig = {"maintenance": f'{m["tail_number"]} {m["aircraft_type"]} {nd["category"]} {tag}'}
            if bundle.safety["no_fly_now"]:
                pillars.append("weather")
                sig["weather"] = "active no-fly window - natural downtime to absorb maintenance"
            cands.append(Candidate(nxt(), "schedule_maintenance", "maintenance",
                                   pillars, sig, frozenset(), horizon))

    # demand: surge -> prestage (requires flight ops)
    for d in bundle.demand:
        if not d["sufficient_history"] or not d["points"]:
            continue
        n = len(d["points"])
        avg = sum(p["forecast"] for p in d["points"]) / n
        lo = sum(p["lower"] for p in d["points"]) / n
        hi = sum(p["upper"] for p in d["points"]) / n
        if avg >= SURGE_PER_DAY:
            sig = {"demand": f'{d["name"]} {avg:.1f}/day [{lo:.1f},{hi:.1f}]'}
            pillars = ["demand"]
            st = _staffing_for_base(bundle, d["base_id"])
            if st and st["crewed_count"] < st["active_aircraft"]:
                pillars.append("staffing")
                sig["staffing"] = f'{st["crewed_count"]} of {st["active_aircraft"]} aircraft crewable'
            cands.append(Candidate(nxt(), "prestage_for_demand", "demand", pillars, sig,
                                   frozenset({"flight"}), 3,
                                   extra={"avg": avg, "width": hi - lo}))

    # staffing gaps -> reposition crew (pool-first, then cross-base) or flag
    for st in bundle.staffing:
        if st["crewed_count"] >= st["active_aircraft"]:
            continue
        for role, short in _short_roles(st).items():
            sig = {"staffing": f'{st["crewed_count"]} of {st["active_aircraft"]} '
                               f'aircraft crewable; short {role} x{short}'}
            # pool-first: reassign locally available crew. Gated if none rested.
            cands.append(Candidate(nxt(), "reposition_crew", "staffing", ["staffing"],
                                   dict(sig), frozenset({"local_rested_crew"}), 2,
                                   extra={"role": role, "mode": "pool"}))
            # fallback: pull from a donor region, else flag it
            donor = gctx.role_donor(role, exclude=bundle.region)
            if donor:
                cost = great_circle_nm(rbase, donor["base"])
                sig2 = dict(sig)
                sig2["logistics"] = f'reposition {role} from {donor["region"]} (~{cost:.0f} nm)'
                cands.append(Candidate(nxt(), "reposition_crew", "staffing",
                                       ["staffing"], sig2, frozenset({"flight"}), 3,
                                       logistics_cost_nm=cost,
                                       extra={"role": role, "mode": "cross_base",
                                              "donor": donor["region"]}))
            else:
                cands.append(Candidate(nxt(), "flag_unresolved_gap", "staffing",
                                       ["staffing"], dict(sig), frozenset(), 0,
                                       extra={"role": role}))

    # capacity: surge with zero crewable aircraft -> reposition an aircraft in
    for d in bundle.demand:
        st = _staffing_for_base(bundle, d["base_id"])
        if not st or st["crewed_count"] != 0 or st["active_aircraft"] == 0:
            continue
        donor = gctx.capacity_donor(exclude=bundle.region)
        if donor:
            cost = great_circle_nm(rbase, donor["base"])
            sig = {"staffing": f'0 of {st["active_aircraft"]} aircraft crewable',
                   "logistics": f'reposition aircraft from {donor["region"]} (~{cost:.0f} nm)'}
            cands.append(Candidate(nxt(), "reposition_aircraft", "staffing",
                                   ["staffing", "demand"], sig, frozenset({"flight"}), 3,
                                   logistics_cost_nm=cost))

    return cands


# -- safety gate (deterministic, runs BEFORE the LLM) ------------------------

def safety_gate(candidates, bundle):
    """Remove infeasible/unsafe options. These are gates, not trade-offs — the
    LLM never gets to weigh safety against operational value."""
    survivors, blocked = [], []
    for c in candidates:
        reason = None
        if "flight" in c.requires and bundle.safety["no_fly_now"]:
            reason = "active no-fly window"
        elif "local_rested_crew" in c.requires and c.extra.get("role") in bundle.safety["rest_gaps"]:
            reason = "no rested crew available in pool"
        if reason:
            c.gate_reason = reason
            blocked.append(c)
        else:
            survivors.append(c)
    return survivors, blocked


# -- ranking via Cortex COMPLETE ---------------------------------------------

def _build_prompt(bundle, survivors):
    def block(label, rows):
        return f"{label}:\n" + ("\n".join(f"  - {r}" for r in rows) if rows else "  (none)")

    signals = "\n".join([
        block("maintenance", [s for c in survivors for s in [c.supporting_signals.get("maintenance")] if s]),
        block("weather", [w.get("next_no_fly") and
                          f'{w["name"]}: {w["no_fly_hours"]}h no-fly, {w["flyable_hours"]}h flyable'
                          or f'{w["name"]}: {w["flyable_hours"]}h flyable' for w in bundle.weather]),
        block("demand", [c.supporting_signals.get("demand") for c in survivors if c.supporting_signals.get("demand")]),
        block("staffing", [c.supporting_signals.get("staffing") for c in survivors if c.supporting_signals.get("staffing")]),
    ])

    safety = bundle.safety
    safety_txt = []
    if safety["no_fly_now"]:
        safety_txt.append("active no-fly weather window now")
    if safety["groundings"]:
        safety_txt.append(f'grounded aircraft: {", ".join(safety["groundings"])}')
    if safety["rest_gaps"]:
        safety_txt.append(f'no rested crew for: {", ".join(safety["rest_gaps"])}')

    cand_lines = []
    for c in survivors:
        sig = "; ".join(f"{k}={v}" for k, v in c.supporting_signals.items())
        cand_lines.append(f'[{c.cid}] action={c.action} pillars={"/".join(c.pillars)} | {sig}')

    return f"""You are the operations orchestrator for a CIVILIAN air-medical and cargo transport fleet (helicopters and light fixed-wing aircraft). Civilian only — never reference military aircraft, ranks, or mission types.

Region: {bundle.region}. Bases: {", ".join(b["name"] for b in bundle.bases)}.

The four specialist signals for this region (grounded data — cite these, invent nothing):
{signals}

SAFETY CONTEXT (hard constraints, already enforced — do not propose anything that violates them):
{chr(10).join("  - " + s for s in safety_txt) if safety_txt else "  (none)"}

FEASIBLE CANDIDATE ACTIONS (these already passed the safety gate; choose ONLY from this list, by id):
{chr(10).join(cand_lines)}

Rank these into a prioritized decision list, best first. Rank by operational value
(demand served, gap severity, revenue protection). Weather-driven actions far out are
low-confidence (weather decays fast) and should rank lower. On ties, prefer the
safety-conservative option. Cite the specific signals above in each rationale.

Return ONLY a JSON array, ordered best-first, one object per candidate you include:
[{{"candidate_id": <id>, "priority_rank": <int>, "horizon_days": <int>, "pillars_addressed": ["..."], "rationale": "<2-3 sentences citing the signals>"}}]"""


def _parse_llm(raw):
    if not raw:
        return []
    s = raw.strip()
    if "```" in s:
        parts = s.split("```")
        if len(parts) >= 2:
            s = parts[1]
            if s.lstrip().lower().startswith("json"):
                s = s.lstrip()[4:]
    a, b = s.find("["), s.rfind("]")
    if a != -1 and b > a:
        try:
            return json.loads(s[a:b + 1])
        except (json.JSONDecodeError, ValueError):
            pass
    return []


def _confidence_for(c, horizon_days):
    if c.primary_pillar == "demand":
        return demand_confidence(horizon_days, c.extra.get("avg", 0), c.extra.get("width", 0))
    return horizon_confidence(c.primary_pillar, horizon_days)


def _package(region, c, rank, horizon_days, pillars, rationale):
    return DecisionPackage(
        region=region, action=c.action, pillars_addressed=pillars or c.pillars,
        priority_rank=rank, horizon_days=horizon_days,
        confidence=_confidence_for(c, horizon_days),
        supporting_signals=c.supporting_signals, rationale=rationale,
        logistics_cost_nm=c.logistics_cost_nm, feedback=None,
    )


def rank_with_llm(bundle, survivors, complete_fn):
    if not survivors:
        return []
    items = _parse_llm(complete_fn(_build_prompt(bundle, survivors)))
    by_cid = {c.cid: c for c in survivors}
    packages, used = [], set()
    for it in items:
        c = by_cid.get(it.get("candidate_id"))
        if not c or c.cid in used:
            continue
        used.add(c.cid)
        horizon = int(it.get("horizon_days", c.default_horizon_days))
        packages.append(_package(bundle.region, c, it.get("priority_rank", len(packages) + 1),
                                 horizon, it.get("pillars_addressed"), it.get("rationale", "")))
    # never silently drop a survivor the model ignored — append in code order
    for c in survivors:
        if c.cid not in used:
            packages.append(_package(bundle.region, c, len(packages) + 1,
                                     c.default_horizon_days, c.pillars,
                                     "(not ranked by model; included by default)"))
    packages.sort(key=lambda p: p.priority_rank)
    return [replace(p, priority_rank=i + 1) for i, p in enumerate(packages)]


def plan_region(bundle, gctx, complete_fn):
    """Full per-region pipeline: candidates -> gate -> rank. Returns
    (packages, blocked) — blocked is what the safety gate removed."""
    survivors, blocked = safety_gate(generate_candidates(bundle, gctx), bundle)
    return rank_with_llm(bundle, survivors, complete_fn), blocked


# -- Snowflake I/O -----------------------------------------------------------

def _get_bases(cur):
    cur.execute("SELECT base_id, name, region, lat, lon FROM bases ORDER BY base_id")
    return {r[0]: {"name": r[1], "region": r[2], "lat": float(r[3]), "lon": float(r[4])}
            for r in cur.fetchall()}


def _get_aircraft_base(cur):
    cur.execute("SELECT aircraft_id, home_base_id FROM aircraft")
    return {r[0]: r[1] for r in cur.fetchall()}


def _make_cortex_fn(cur):
    def _fn(prompt):
        cur.execute("SELECT SNOWFLAKE.CORTEX.COMPLETE(%s, %s)", (CORTEX_MODEL, prompt))
        return cur.fetchone()[0]
    return _fn


def _ensure_table(cur):
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {DECISION_TABLE} (
            package_id INTEGER AUTOINCREMENT,
            region VARCHAR, action VARCHAR, priority_rank INTEGER,
            horizon_days INTEGER, confidence FLOAT,
            pillars_addressed VARIANT, supporting_signals VARIANT,
            rationale VARCHAR, logistics_cost_nm FLOAT,
            generated_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
            feedback VARIANT
        )
    """)


def _persist(cur, packages):
    for p in packages:
        cur.execute(f"""
            INSERT INTO {DECISION_TABLE}
                (region, action, priority_rank, horizon_days, confidence,
                 pillars_addressed, supporting_signals, rationale,
                 logistics_cost_nm, generated_at, feedback)
            SELECT %s, %s, %s, %s, %s, PARSE_JSON(%s), PARSE_JSON(%s), %s, %s,
                   CURRENT_TIMESTAMP(), NULL
        """, (p.region, p.action, p.priority_rank, p.horizon_days, p.confidence,
              json.dumps(p.pillars_addressed), json.dumps(p.supporting_signals),
              p.rationale, p.logistics_cost_nm))


# -- public API --------------------------------------------------------------

def synthesize(state, conn=None, complete_fn=None) -> list[DecisionPackage]:
    """Synthesize per-region decision packages from the four agents' state,
    persist them, and return them."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    cur = conn.cursor()
    try:
        bases = _get_bases(cur)
        aircraft_base = _get_aircraft_base(cur)
        bundles = build_bundles(state, bases, aircraft_base)
        gctx = build_global_ctx(bundles)
        complete_fn = complete_fn or _make_cortex_fn(cur)

        packages = []
        for b in bundles:
            pkgs, _blocked = plan_region(b, gctx, complete_fn)
            packages.extend(pkgs)

        _ensure_table(cur)
        _persist(cur, packages)
        conn.commit()
        return packages
    finally:
        cur.close()
        if own_conn:
            conn.close()


def to_dict(packages: list[DecisionPackage]) -> list[dict]:
    return [
        {
            "region": p.region, "action": p.action,
            "pillars_addressed": p.pillars_addressed,
            "priority_rank": p.priority_rank, "horizon_days": p.horizon_days,
            "confidence": p.confidence, "supporting_signals": p.supporting_signals,
            "rationale": p.rationale, "logistics_cost_nm": p.logistics_cost_nm,
            "feedback": p.feedback,
        }
        for p in packages
    ]


def print_report(packages: list[DecisionPackage]):
    print(f"\n{'='*78}")
    print(f"DECISION PACKAGES - {len(packages)} across "
          f"{len({p.region for p in packages})} regions")
    print(f"{'='*78}")
    for region in sorted({p.region for p in packages}):
        print(f"\n  {region}")
        for p in sorted([x for x in packages if x.region == region], key=lambda x: x.priority_rank):
            cost = f" ~{p.logistics_cost_nm:.0f}nm" if p.logistics_cost_nm else ""
            print(f"    {p.priority_rank}. {p.action:<22} conf={p.confidence:<5} "
                  f"h={p.horizon_days}d{cost}  [{'/'.join(p.pillars_addressed)}]")
            print(f"       signals: {p.supporting_signals}")
            print(f"       why: {p.rationale}")


def _print_dicts(recs):
    print(f"\n{'='*78}")
    print(f"DECISION PACKAGES - {len(recs)} across {len({r['region'] for r in recs})} regions")
    print(f"{'='*78}")
    for region in sorted({r["region"] for r in recs}):
        print(f"\n  {region}")
        for p in sorted([x for x in recs if x["region"] == region], key=lambda x: x["priority_rank"]):
            cost = f" ~{p['logistics_cost_nm']:.0f}nm" if p["logistics_cost_nm"] else ""
            print(f"    {p['priority_rank']}. {p['action']:<22} conf={p['confidence']:<5} "
                  f"h={p['horizon_days']}d{cost}  [{'/'.join(p['pillars_addressed'])}]")
            print(f"       signals: {p['supporting_signals']}")
            print(f"       why: {p['rationale']}")


if __name__ == "__main__":
    # the graph's orchestrator node already synthesizes — just run and print.
    from graph.workflow import run
    final = run()
    _print_dicts(final.get("recommendations") or [])
