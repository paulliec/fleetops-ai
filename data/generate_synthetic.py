"""Generate synthetic fleet operations data for development and testing.

Produces CSV files matching the schema in schema.sql.
50 aircraft, 6 bases, ~2 years of history.

Usage:
    python -m data.generate_synthetic          # writes to data/output/
    python -m data.generate_synthetic --seed 42
"""

import argparse
import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"

# -- seed data ---------------------------------------------------------------

BASES = [
    (1, "Southeast Medevac Center", "KPDK", 33.88, -84.30, "Southeast", "US/Eastern"),
    (2, "Mountain West Hub",        "KAPA", 39.57, -104.85, "Mountain", "US/Mountain"),
    (3, "Midwest Logistics Center", "KMKC", 39.12, -94.59, "Midwest", "US/Central"),
    (4, "Northeast Air Operations", "KTEB", 40.85, -74.06, "Northeast", "US/Eastern"),
    (5, "Southwest Regional Hub",   "KDVT", 33.69, -112.08, "Southwest", "US/Arizona"),
    (6, "Gulf Coast Operations",    "KHOU", 29.65, -95.28, "Gulf Coast", "US/Central"),
]

# type -> (category, engine_interval_hrs, airframe_interval_hrs, avionics_check_days, typical_speed_kts)
AIRCRAFT_TYPES = {
    "King Air 350":  ("fixed_wing", 1500, 3000, 180, 310),
    "PC-12":         ("fixed_wing", 1200, 2400, 180, 270),
    "Citation CJ3":  ("fixed_wing", 1800, 3500, 180, 415),
    "Bell 407":      ("rotary",      800, 1500, 120, 140),
    "EC135":         ("rotary",      600, 1200, 120, 135),
    "AW139":         ("rotary",      900, 1800, 120, 165),
}

# distribution of 50 aircraft across types
AIRCRAFT_DISTRIBUTION = {
    "King Air 350": 10, "PC-12": 8, "Citation CJ3": 6,
    "Bell 407": 10, "EC135": 8, "AW139": 8,
}

RANKS = ["Captain", "First Officer", "Senior Captain", "Check Airman",
         "Line Pilot", "Chief Pilot"]
ROLES = ["pilot", "copilot", "flight_nurse", "flight_paramedic"]

# role -> which aircraft types they can be qualified on
ROLE_AIRCRAFT = {
    "pilot":            list(AIRCRAFT_TYPES.keys()),
    "copilot":          list(AIRCRAFT_TYPES.keys()),
    "flight_nurse":     ["Bell 407", "EC135", "AW139", "King Air 350"],
    "flight_paramedic": ["Bell 407", "EC135", "AW139"],
}

MISSION_TYPES = ["cargo", "medevac", "organ_transport", "charter", "repositioning"]
PRIORITIES = ["routine", "urgent", "critical"]
PRIORITY_WEIGHTS = [0.65, 0.25, 0.10]

# per-base average daily mission count — varies by regional demand
# total ~35/day across all bases, ~25,500 over 2 years
BASE_DAILY_MISSIONS = {
    1: (5, 7),    # Southeast Medevac — steady trauma/transfer volume
    2: (4, 6),    # Mountain West — lower density, longer hauls
    3: (5, 7),    # Midwest Logistics — central cargo hub
    4: (6, 8),    # Northeast — dense metro, high medevac + cargo
    5: (5, 7),    # Southwest — growing regional demand
    6: (8, 10),   # Gulf Coast — highest volume, oil/gas + hurricane corridor
}

# seasonal demand multipliers by region (month -> multiplier)
# baked-in signal the demand agent should be able to recover
import math

def seasonal_multiplier(base_id, month):
    """Regional seasonal patterns for mission volume."""
    if base_id == 6:  # Gulf Coast — hurricane season Jun-Nov spikes demand
        return 1.0 + 0.4 * math.sin(math.pi * (month - 3) / 6) if 6 <= month <= 11 else 0.85
    if base_id == 2:  # Mountain — winter drops, summer peak
        return 0.7 + 0.5 * math.sin(math.pi * (month - 1) / 11)
    if base_id == 4:  # Northeast — winter dip from weather
        return 0.75 + 0.4 * math.sin(math.pi * (month - 1) / 11)
    if base_id == 3:  # Midwest — mild seasonal, harvest season bump
        return 0.9 + 0.2 * math.sin(math.pi * (month - 3) / 6)
    if base_id == 1:  # Southeast — slight winter uptick (flu season transfers)
        return 1.0 + 0.15 * math.cos(math.pi * (month - 1) / 6)
    if base_id == 5:  # Southwest — summer heat dip
        return 1.05 - 0.2 * math.sin(math.pi * (month - 3) / 6)
    return 1.0


# linear trend: ~2% annual growth in demand
def trend_multiplier(day_offset):
    """Slow upward trend over the 2-year window."""
    return 1.0 + 0.04 * (day_offset / 365)

PRECIP_TYPES = ["none", "rain", "snow", "ice", "fog"]
PRECIP_INTENSITY = [None, "light", "moderate", "heavy"]

DELAY_REASONS = ["weather", "maintenance", "crew", "airspace"]

MAINT_CATEGORIES = ["engine", "avionics", "airframe", "hydraulic", "landing_gear"]
MAINT_CATEGORIES_ROTARY = ["engine", "avionics", "airframe", "hydraulic", "rotor"]

FIRST_NAMES = [
    "James", "Maria", "Robert", "Sarah", "Michael", "Jennifer", "David", "Emily",
    "Carlos", "Aisha", "Kevin", "Lisa", "Brian", "Samantha", "Daniel", "Nicole",
    "Marcus", "Rachel", "Tyler", "Angela", "Jason", "Michelle", "Ryan", "Jessica",
    "Andre", "Kimberly", "Nathan", "Laura", "Sean", "Megan", "Eric", "Amanda",
    "Chris", "Stephanie", "Travis", "Diana", "Kyle", "Tiffany", "Derek", "Heather",
]
LAST_NAMES = [
    "Johnson", "Williams", "Chen", "Garcia", "Brown", "Martinez", "Davis", "Wilson",
    "Anderson", "Thomas", "Jackson", "White", "Harris", "Clark", "Lewis", "Robinson",
    "Walker", "Young", "Allen", "King", "Wright", "Lopez", "Hill", "Scott",
    "Green", "Adams", "Baker", "Hall", "Rivera", "Campbell", "Mitchell", "Carter",
]

# -- time range ---------------------------------------------------------------

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 12, 31)
TOTAL_DAYS = (END_DATE - START_DATE).days


def random_dt(start=START_DATE, end=END_DATE):
    delta = (end - start).total_seconds()
    return start + timedelta(seconds=random.random() * delta)


# -- generators ---------------------------------------------------------------

def gen_bases():
    return [
        dict(base_id=b[0], name=b[1], icao_code=b[2], lat=b[3], lon=b[4],
             region=b[5], timezone=b[6])
        for b in BASES
    ]


# N-number suffixes by type for realistic civilian tail numbers
_TAIL_PREFIX = {
    "King Air 350": "KB", "PC-12": "PC", "Citation CJ3": "CJ",
    "Bell 407": "BH", "EC135": "EC", "AW139": "AW",
}


def gen_aircraft():
    aircraft = []
    aid = 1
    for atype, count in AIRCRAFT_DISTRIBUTION.items():
        cat, eng_int, af_int, av_days, _ = AIRCRAFT_TYPES[atype]
        prefix = _TAIL_PREFIX[atype]
        for i in range(count):
            base = random.choice(BASES)
            commissioned = START_DATE - timedelta(days=random.randint(365, 3650))
            status = random.choices(["active", "grounded", "depot"], [0.82, 0.12, 0.06])[0]
            aircraft.append(dict(
                aircraft_id=aid,
                tail_number=f"N{100 + aid}{prefix}",
                aircraft_type=atype,
                category=cat,
                home_base_id=base[0],
                status=status,
                commissioned_date=commissioned.strftime("%Y-%m-%d"),
                total_flight_hours=0.0,  # derived from flight logs after generation
                engine_service_interval_hours=eng_int,
                airframe_inspection_interval_hours=af_int,
                avionics_check_interval_days=av_days,
            ))
            aid += 1
    return aircraft


def gen_crew(aircraft_list):
    """~120 crew across all bases, qualified on aircraft at their base."""
    crew = []
    base_aircraft = {}
    for a in aircraft_list:
        base_aircraft.setdefault(a["home_base_id"], set()).add(a["aircraft_type"])

    cid = 1
    for base in BASES:
        bid = base[0]
        types_at_base = list(base_aircraft.get(bid, list(AIRCRAFT_TYPES.keys())[:2]))
        num_crew = random.randint(16, 24)
        for _ in range(num_crew):
            role = random.choice(ROLES)
            eligible = [t for t in types_at_base if t in ROLE_AIRCRAFT.get(role, [])]
            if not eligible:
                eligible = [random.choice(types_at_base)]
            quals = random.sample(eligible, min(len(eligible), random.randint(1, 3)))
            name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
            status = random.choices(
                ["available", "on_assignment", "on_leave", "medical"],
                [0.70, 0.15, 0.10, 0.05]
            )[0]
            crew.append(dict(
                crew_id=cid,
                name=name,
                rank=random.choice(RANKS),
                role=role,
                base_id=bid,
                status=status,
                qualifications=str(quals),  # stored as string repr of list for CSV
                total_flight_hours=round(random.uniform(200, 5000), 1),
            ))
            cid += 1
    return crew


def gen_missions(aircraft_list):
    """~25k missions over 2 years, per-base daily rates from BASE_DAILY_MISSIONS."""
    missions = []
    # index active aircraft by home base for realistic assignment
    active_by_base = {}
    for a in aircraft_list:
        if a["status"] == "active":
            active_by_base.setdefault(a["home_base_id"], []).append(a)
    all_active = [a for a in aircraft_list if a["status"] == "active"]
    mid = 1

    for day_offset in range(TOTAL_DAYS):
        date = START_DATE + timedelta(days=day_offset)
        trend = trend_multiplier(day_offset)
        for base in BASES:
            bid = base[0]
            lo, hi = BASE_DAILY_MISSIONS[bid]
            base_mean = (lo + hi) / 2
            adjusted = base_mean * seasonal_multiplier(bid, date.month) * trend
            n_missions = max(1, int(random.gauss(adjusted, 1.5)))

            for _ in range(n_missions):
                mtype = random.choice(MISSION_TYPES)
                priority = random.choices(PRIORITIES, PRIORITY_WEIGHTS)[0]
                # repositioning stays at home base
                dest = None if mtype == "repositioning" else random.choice(
                    [b for b in BASES if b[0] != bid]
                )
                # prefer aircraft at this base, fall back to any active
                pool = active_by_base.get(bid, all_active)
                aircraft = random.choice(pool) if random.random() > 0.05 else None
                req_dt = date + timedelta(hours=random.randint(0, 23), minutes=random.randint(0, 59))

                is_past = date < datetime(2025, 12, 1)
                if is_past:
                    start_dt = req_dt + timedelta(hours=random.randint(0, 4))
                    duration_hrs = random.uniform(1, 12)
                    end_dt = start_dt + timedelta(hours=duration_hrs)
                    status = random.choices(["complete", "cancelled"], [0.90, 0.10])[0]
                else:
                    start_dt = None
                    end_dt = None
                    status = random.choices(["planned", "active"], [0.6, 0.4])[0]

                pax = random.randint(1, 8) if mtype in ("medevac", "organ_transport", "charter") else None
                cargo = round(random.uniform(0.5, 20), 1) if mtype == "cargo" else None

                missions.append(dict(
                    mission_id=mid,
                    mission_type=mtype,
                    priority=priority,
                    requested_date=req_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    start_date=start_dt.strftime("%Y-%m-%d %H:%M:%S") if start_dt else None,
                    end_date=end_dt.strftime("%Y-%m-%d %H:%M:%S") if end_dt else None,
                    origin_base_id=bid,
                    dest_base_id=dest[0] if dest else None,
                    aircraft_id=aircraft["aircraft_id"] if aircraft else None,
                    status=status,
                    pax_count=pax,
                    cargo_tons=cargo,
                ))
                mid += 1
    return missions


def _make_leg(fid, mission, aircraft_id, pic, origin, dest, sched_dep, flight_hrs, is_complete):
    """Build one flight log row."""
    sched_arr = sched_dep + timedelta(hours=flight_hrs)

    delay = 0
    delay_reason = None
    if random.random() < 0.20:
        delay = random.choices(
            [15, 30, 60, 120, 240],
            [0.35, 0.30, 0.20, 0.10, 0.05]
        )[0]
        delay_reason = random.choice(DELAY_REASONS)

    actual_dep = sched_dep + timedelta(minutes=delay)
    actual_arr = actual_dep + timedelta(hours=flight_hrs)

    if is_complete:
        fl_status = "completed"
        if delay_reason == "weather" and random.random() < 0.08:
            fl_status = "diverted"
    else:
        fl_status = "airborne"

    fuel = round(flight_hrs * random.uniform(800, 4000), 0)

    return dict(
        flight_id=fid,
        mission_id=mission["mission_id"],
        aircraft_id=aircraft_id,
        pic_crew_id=pic,
        origin_base_id=origin,
        dest_base_id=dest,
        scheduled_departure=sched_dep.strftime("%Y-%m-%d %H:%M:%S"),
        actual_departure=actual_dep.strftime("%Y-%m-%d %H:%M:%S"),
        scheduled_arrival=sched_arr.strftime("%Y-%m-%d %H:%M:%S"),
        actual_arrival=actual_arr.strftime("%Y-%m-%d %H:%M:%S") if fl_status in ("completed", "diverted") else None,
        flight_hours=round(flight_hrs, 1),
        fuel_burn_lbs=fuel,
        status=fl_status,
        delay_minutes=delay,
        delay_reason=delay_reason,
    ), actual_arr


def gen_flight_logs(missions, crew_list):
    """1-2 flight legs per assigned mission. Cross-base missions get a return leg ~40% of the time."""
    logs = []
    fid = 1
    pilots_by_base = {}
    for c in crew_list:
        if c["role"] in ("pilot", "copilot"):
            pilots_by_base.setdefault(c["base_id"], []).append(c["crew_id"])

    for m in missions:
        if m["aircraft_id"] is None:
            continue
        if m["status"] not in ("complete", "active"):
            continue

        origin = m["origin_base_id"]
        dest = m["dest_base_id"] or origin
        pilots = pilots_by_base.get(origin, pilots_by_base.get(1, [1]))
        pic = random.choice(pilots)
        is_complete = m["status"] == "complete"

        sched_dep = datetime.strptime(m["requested_date"], "%Y-%m-%d %H:%M:%S")
        flight_hrs = random.uniform(1, 10)

        # outbound leg
        leg, arr_time = _make_leg(fid, m, m["aircraft_id"], pic, origin, dest, sched_dep, flight_hrs, is_complete)
        logs.append(leg)
        fid += 1

        # return leg for cross-base missions (~40%)
        has_return = (dest != origin) and random.random() < 0.40
        if has_return and is_complete:
            turnaround = timedelta(hours=random.uniform(1, 6))
            return_dep = arr_time + turnaround
            return_hrs = flight_hrs * random.uniform(0.85, 1.15)  # slight variation
            return_pilots = pilots_by_base.get(dest, pilots)
            leg2, _ = _make_leg(fid, m, m["aircraft_id"], random.choice(return_pilots),
                                dest, origin, return_dep, return_hrs, is_complete)
            logs.append(leg2)
            fid += 1

    return logs


def gen_maintenance(aircraft_list, flight_logs):
    """Maintenance events derived from flight hour accumulation.

    Walks each aircraft's flight logs chronologically, triggers scheduled
    maintenance at interval thresholds, plus random unscheduled events
    proportional to flight activity.
    """
    events = []
    eid = 1

    # accumulate flight hours per aircraft in chronological order
    hours_by_aircraft = {}
    for fl in flight_logs:
        aid = fl["aircraft_id"]
        hours_by_aircraft.setdefault(aid, []).append((fl["scheduled_departure"], fl["flight_hours"]))

    for a in aircraft_list:
        aid = a["aircraft_id"]
        atype_info = AIRCRAFT_TYPES[a["aircraft_type"]]
        cat = atype_info[0]
        eng_interval = atype_info[1]
        af_interval = atype_info[2]
        av_days = atype_info[3]
        categories = MAINT_CATEGORIES_ROTARY if cat == "rotary" else MAINT_CATEGORIES

        logs = hours_by_aircraft.get(aid, [])
        logs.sort(key=lambda x: x[0])

        # track thresholds for scheduled maintenance
        next_engine_due = eng_interval
        next_airframe_due = af_interval
        last_avionics = START_DATE
        cumulative_hrs = 0.0

        for dep_str, hrs in logs:
            dep_dt = datetime.strptime(dep_str, "%Y-%m-%d %H:%M:%S")
            cumulative_hrs += hrs

            # engine service at interval
            if cumulative_hrs >= next_engine_due:
                event_dt = dep_dt + timedelta(hours=random.randint(1, 12))
                dur = random.randint(8, 48)
                completed = event_dt + timedelta(hours=dur)
                grounded = random.choices([1, 2, 3, 5], [0.3, 0.4, 0.2, 0.1])[0]
                events.append(dict(
                    event_id=eid, aircraft_id=aid, base_id=a["home_base_id"],
                    event_type="scheduled", category="engine",
                    started_at=event_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    completed_at=completed.strftime("%Y-%m-%d %H:%M:%S") if completed < END_DATE else None,
                    aircraft_hours_at_event=round(cumulative_hrs, 1),
                    next_due_hours=round(cumulative_hrs + eng_interval, 1),
                    next_due_date=(event_dt + timedelta(days=av_days)).strftime("%Y-%m-%d"),
                    grounded_days=grounded,
                    description="scheduled engine service",
                ))
                eid += 1
                next_engine_due = cumulative_hrs + eng_interval

            # airframe inspection at interval
            if cumulative_hrs >= next_airframe_due:
                event_dt = dep_dt + timedelta(hours=random.randint(1, 24))
                dur = random.randint(4, 24)
                completed = event_dt + timedelta(hours=dur)
                events.append(dict(
                    event_id=eid, aircraft_id=aid, base_id=a["home_base_id"],
                    event_type="inspection", category="airframe",
                    started_at=event_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    completed_at=completed.strftime("%Y-%m-%d %H:%M:%S") if completed < END_DATE else None,
                    aircraft_hours_at_event=round(cumulative_hrs, 1),
                    next_due_hours=round(cumulative_hrs + af_interval, 1),
                    next_due_date=None,
                    grounded_days=random.choices([0, 1, 2], [0.5, 0.3, 0.2])[0],
                    description="scheduled airframe inspection",
                ))
                eid += 1
                next_airframe_due = cumulative_hrs + af_interval

            # avionics check by calendar interval
            if (dep_dt - last_avionics).days >= av_days:
                event_dt = dep_dt + timedelta(hours=random.randint(1, 8))
                dur = random.randint(2, 12)
                completed = event_dt + timedelta(hours=dur)
                events.append(dict(
                    event_id=eid, aircraft_id=aid, base_id=a["home_base_id"],
                    event_type="inspection", category="avionics",
                    started_at=event_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    completed_at=completed.strftime("%Y-%m-%d %H:%M:%S") if completed < END_DATE else None,
                    aircraft_hours_at_event=round(cumulative_hrs, 1),
                    next_due_hours=None,
                    next_due_date=(event_dt + timedelta(days=av_days)).strftime("%Y-%m-%d"),
                    grounded_days=0,
                    description="scheduled avionics check",
                ))
                eid += 1
                last_avionics = dep_dt

            # unscheduled events — ~3% chance per flight
            if random.random() < 0.03:
                maint_cat = random.choice(categories)
                event_dt = dep_dt + timedelta(hours=random.randint(0, 4))
                grounded = random.choices([0, 1, 2, 3, 5, 10, 20], [0.15, 0.20, 0.20, 0.20, 0.10, 0.10, 0.05])[0]
                dur = random.randint(4, 96)
                completed = event_dt + timedelta(hours=dur)
                events.append(dict(
                    event_id=eid, aircraft_id=aid, base_id=a["home_base_id"],
                    event_type="unscheduled", category=maint_cat,
                    started_at=event_dt.strftime("%Y-%m-%d %H:%M:%S"),
                    completed_at=completed.strftime("%Y-%m-%d %H:%M:%S") if completed < END_DATE else None,
                    aircraft_hours_at_event=round(cumulative_hrs, 1),
                    next_due_hours=None,
                    next_due_date=None,
                    grounded_days=grounded,
                    description=f"unscheduled {maint_cat} repair",
                ))
                eid += 1

    return events


def gen_weather():
    """Hourly weather observations for each base — sampled to every 3 hours to keep size manageable."""
    obs = []
    oid = 1
    for base in BASES:
        lat = base[3]
        # rough climate adjustments
        is_hot = lat < 30
        is_cold = lat > 45

        for day_offset in range(TOTAL_DAYS):
            date = START_DATE + timedelta(days=day_offset)
            # every 3 hours
            for hour in range(0, 24, 3):
                dt = date + timedelta(hours=hour)
                month = dt.month

                # seasonal temp baseline
                if is_hot:
                    temp = random.gauss(35, 5) if month in (5, 6, 7, 8, 9) else random.gauss(22, 5)
                elif is_cold:
                    temp = random.gauss(2, 8) if month in (11, 12, 1, 2) else random.gauss(18, 6)
                else:
                    temp = random.gauss(12, 10) if month in (11, 12, 1, 2) else random.gauss(22, 6)

                wind = max(0, int(random.gauss(10, 6)))
                gust = wind + random.randint(0, 15) if random.random() < 0.3 else None
                vis = round(max(0.25, random.gauss(8, 3)), 1)
                ceiling = max(200, int(random.gauss(15000, 8000)))

                # precip
                if random.random() < 0.15:
                    if temp < 0:
                        precip = random.choice(["snow", "ice"])
                    elif vis < 3:
                        precip = "fog"
                    else:
                        precip = "rain"
                    intensity = random.choice(["light", "moderate", "heavy"])
                else:
                    precip = "none"
                    intensity = None

                # flight category based on ceiling + vis
                if ceiling >= 3000 and vis >= 5:
                    fcat = "VFR"
                elif ceiling >= 1000 and vis >= 3:
                    fcat = "MVFR"
                elif ceiling >= 500 and vis >= 1:
                    fcat = "IFR"
                else:
                    fcat = "LIFR"

                obs.append(dict(
                    observation_id=oid,
                    base_id=base[0],
                    observed_at=dt.strftime("%Y-%m-%d %H:%M:%S"),
                    temp_c=round(temp, 1),
                    wind_speed_kts=wind,
                    wind_gust_kts=gust,
                    visibility_sm=vis,
                    ceiling_ft=ceiling,
                    precip_type=precip,
                    precip_intensity=intensity,
                    flight_category=fcat,
                ))
                oid += 1
    return obs


# -- write helpers ------------------------------------------------------------

def write_csv(rows, filename):
    if not rows:
        return
    path = OUTPUT_DIR / filename
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {filename}: {len(rows):,} rows")


# -- main ---------------------------------------------------------------------

def derive_flight_hours(aircraft, flight_logs):
    """Set aircraft.total_flight_hours = sum of its flight log hours."""
    totals = {}
    for fl in flight_logs:
        totals[fl["aircraft_id"]] = totals.get(fl["aircraft_id"], 0.0) + fl["flight_hours"]
    for a in aircraft:
        a["total_flight_hours"] = round(totals.get(a["aircraft_id"], 0.0), 1)


def seed_end_states(aircraft, flight_logs):
    """Plant specific interesting scenarios for agents to find.

    - Aircraft within 50 hrs of engine service interval (adds flight log entries)
    - A rotary aircraft near airframe life limit (adds flight log entries)
    - A grounded aircraft with overdue avionics check
    """
    totals = {}
    for fl in flight_logs:
        totals[fl["aircraft_id"]] = totals.get(fl["aircraft_id"], 0.0) + fl["flight_hours"]

    seeded = []
    seeded_ids = set()
    next_fid = max(fl["flight_id"] for fl in flight_logs) + 1

    # find a high-hours active fixed wing, put it just under engine interval
    for a in aircraft:
        if a["category"] == "fixed_wing" and a["status"] == "active":
            hrs = totals.get(a["aircraft_id"], 0)
            interval = a["engine_service_interval_hours"]
            if hrs > interval:
                remainder = hrs % interval
                gap = interval - remainder
                if gap > 50:
                    # inject flights to close the gap, leaving ~30-50 hrs
                    target_gap = random.uniform(30, 50)
                    needed = gap - target_gap
                    # spread across a few recent flights
                    n_extra = random.randint(3, 6)
                    per_flight = needed / n_extra
                    for i in range(n_extra):
                        dep = datetime(2025, 11, 20 + i, random.randint(6, 18))
                        flight_logs.append(dict(
                            flight_id=next_fid, mission_id=1,  # placeholder ref
                            aircraft_id=a["aircraft_id"], pic_crew_id=1,
                            origin_base_id=a["home_base_id"], dest_base_id=a["home_base_id"],
                            scheduled_departure=dep.strftime("%Y-%m-%d %H:%M:%S"),
                            actual_departure=dep.strftime("%Y-%m-%d %H:%M:%S"),
                            scheduled_arrival=(dep + timedelta(hours=per_flight)).strftime("%Y-%m-%d %H:%M:%S"),
                            actual_arrival=(dep + timedelta(hours=per_flight)).strftime("%Y-%m-%d %H:%M:%S"),
                            flight_hours=round(per_flight, 1),
                            fuel_burn_lbs=round(per_flight * 2000, 0),
                            status="completed", delay_minutes=0, delay_reason=None,
                        ))
                        next_fid += 1
                    new_total = hrs + needed
                    remaining = interval - (new_total % interval)
                    seeded_ids.add(a["aircraft_id"])
                    seeded.append(f"  aircraft {a['aircraft_id']} ({a['tail_number']}): "
                                  f"~{round(remaining)} hrs to engine service")
                    break

    # find a rotary, inject flights to push it to high airframe hours
    for a in aircraft:
        if a["category"] == "rotary" and a["status"] == "active" and a["aircraft_id"] not in seeded_ids:
            current = totals.get(a["aircraft_id"], 0)
            target = a["airframe_inspection_interval_hours"] * 4.8
            needed = target - current
            if needed > 0:
                n_extra = 20
                per_flight = needed / n_extra
                for i in range(n_extra):
                    dep = datetime(2025, 10, 1) + timedelta(days=i * 3)
                    flight_logs.append(dict(
                        flight_id=next_fid, mission_id=1,
                        aircraft_id=a["aircraft_id"], pic_crew_id=1,
                        origin_base_id=a["home_base_id"], dest_base_id=a["home_base_id"],
                        scheduled_departure=dep.strftime("%Y-%m-%d %H:%M:%S"),
                        actual_departure=dep.strftime("%Y-%m-%d %H:%M:%S"),
                        scheduled_arrival=(dep + timedelta(hours=per_flight)).strftime("%Y-%m-%d %H:%M:%S"),
                        actual_arrival=(dep + timedelta(hours=per_flight)).strftime("%Y-%m-%d %H:%M:%S"),
                        flight_hours=round(per_flight, 1),
                        fuel_burn_lbs=round(per_flight * 1500, 0),
                        status="completed", delay_minutes=0, delay_reason=None,
                    ))
                    next_fid += 1
            seeded_ids.add(a["aircraft_id"])
            seeded.append(f"  aircraft {a['aircraft_id']} ({a['tail_number']}): "
                          f"rotary at {round(target)} hrs (near airframe life limit)")
            break

    # ground one rotary with an overdue avionics check
    for a in aircraft:
        if a["status"] == "active" and a["category"] == "rotary" and a["aircraft_id"] not in seeded_ids:
            a["status"] = "grounded"
            seeded.append(f"  aircraft {a['aircraft_id']} ({a['tail_number']}): "
                          f"grounded, overdue avionics check")
            break

    if seeded:
        print("  Seeded end states:")
        for s in seeded:
            print(s)


def main(seed=42):
    random.seed(seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic fleet data...")

    bases = gen_bases()
    write_csv(bases, "bases.csv")

    aircraft = gen_aircraft()

    crew = gen_crew(aircraft)
    write_csv(crew, "crew.csv")

    missions = gen_missions(aircraft)
    write_csv(missions, "missions.csv")

    flight_logs = gen_flight_logs(missions, crew)

    # seed edge cases (may inject extra flight log rows), then derive hours
    seed_end_states(aircraft, flight_logs)
    derive_flight_hours(aircraft, flight_logs)
    write_csv(flight_logs, "flight_logs.csv")
    write_csv(aircraft, "aircraft.csv")

    maintenance = gen_maintenance(aircraft, flight_logs)
    write_csv(maintenance, "maintenance_events.csv")

    weather = gen_weather()
    write_csv(weather, "weather_observations.csv")

    print(f"\nDone. Files written to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic fleet ops data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    main(seed=args.seed)
