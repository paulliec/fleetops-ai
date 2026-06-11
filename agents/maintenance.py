"""Maintenance forecasting — standalone module, no LLM or LangGraph.

Queries Snowflake for active aircraft, flight activity, and maintenance
history. Forecasts the next due maintenance per aircraft across four
trigger types: hours-based, calendar, dual (whichever-comes-first),
and life-limited. Surfaces tempo anomalies and consolidation opportunities.

Usage:
    python -m agents.maintenance
"""

from dataclasses import dataclass, field
from datetime import date, timedelta

from utils.snowflake import get_connection

CONSOLIDATION_WINDOW_DAYS = 14
TEMPO_THRESHOLD = 1.3  # 30d rate > 1.3x 90d rate = flag
AIRFRAME_CALENDAR_DAYS = 365  # dual trigger: hours OR 1 year

# hard airframe life limits (hours) — rotary types
LIFE_LIMITS = {
    "Bell 407": 10_000,
    "EC135":    12_000,
    "AW139":    15_000,
}


@dataclass
class DueItem:
    category: str       # engine | airframe | avionics | airframe_life
    trigger: str        # hours | calendar | dual | life_limit
    due_at_hours: float | None
    due_date: date | None
    hours_remaining: float | None
    days_remaining: int | None
    hard_ground: bool


@dataclass
class AircraftForecast:
    aircraft_id: int
    tail_number: str
    aircraft_type: str
    total_hours: float
    rate_30d: float
    rate_90d: float
    tempo_flag: bool
    next_due: DueItem
    consolidation: list[DueItem] = field(default_factory=list)


# -- data fetching -----------------------------------------------------------

def _get_as_of(cur):
    cur.execute("SELECT MAX(scheduled_departure)::DATE FROM flight_logs")
    return cur.fetchone()[0]


def _get_aircraft(cur):
    cur.execute("""
        SELECT aircraft_id, tail_number, aircraft_type, category,
               total_flight_hours, engine_service_interval_hours,
               airframe_inspection_interval_hours, avionics_check_interval_days
        FROM aircraft WHERE status = 'active'
        ORDER BY aircraft_id
    """)
    cols = [d[0].lower() for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _get_flight_rates(cur, as_of):
    """30-day and 90-day rolling flight rate (hrs/day) per aircraft."""
    cur.execute("""
        SELECT aircraft_id,
            COALESCE(SUM(CASE WHEN scheduled_departure >= DATEADD(day, -30, %(d)s)
                         THEN flight_hours END), 0) / 30.0,
            COALESCE(SUM(CASE WHEN scheduled_departure >= DATEADD(day, -90, %(d)s)
                         THEN flight_hours END), 0) / 90.0
        FROM flight_logs
        WHERE status IN ('completed', 'airborne')
        GROUP BY aircraft_id
    """, {"d": as_of})
    return {r[0]: (float(r[1]), float(r[2])) for r in cur.fetchall()}


def _get_last_service(cur):
    """Last scheduled/inspection service point per aircraft for each tracked category.

    Only scheduled + inspection events reset the interval clock.
    Unscheduled repairs don't.
    """
    cur.execute("""
        SELECT aircraft_id,
            MAX(CASE WHEN category = 'engine' THEN aircraft_hours_at_event END),
            MAX(CASE WHEN category = 'airframe' THEN aircraft_hours_at_event END),
            MAX(CASE WHEN category = 'airframe' THEN started_at END)::DATE,
            MAX(CASE WHEN category = 'avionics' THEN started_at END)::DATE
        FROM maintenance_events
        WHERE completed_at IS NOT NULL
          AND event_type IN ('scheduled', 'inspection')
        GROUP BY aircraft_id
    """)
    result = {}
    for r in cur.fetchall():
        result[r[0]] = {
            "engine_hrs": float(r[1]) if r[1] else None,
            "airframe_hrs": float(r[2]) if r[2] else None,
            "airframe_date": r[3],
            "avionics_date": r[4],
        }
    return result


# -- forecasting logic -------------------------------------------------------

def _hrs_to_date(hrs_remaining, rate, as_of):
    if rate <= 0 or hrs_remaining is None:
        return None
    return as_of + timedelta(days=int(hrs_remaining / rate))


def _hrs_to_days(hrs_remaining, rate):
    if rate <= 0 or hrs_remaining is None:
        return None
    return int(hrs_remaining / rate)


def _forecast_one(ac, rates, last_service, as_of):
    aid = ac["aircraft_id"]
    total_hrs = float(ac["total_flight_hours"])
    rate_30d, rate_90d = rates.get(aid, (0.0, 0.0))
    rate = rate_30d if rate_30d > 0 else rate_90d
    svc = last_service.get(aid, {})

    items = []

    # --- engine: hours-based ---
    eng_interval = float(ac["engine_service_interval_hours"])
    last_eng = svc.get("engine_hrs")
    eng_due = (last_eng + eng_interval) if last_eng else eng_interval
    eng_rem = eng_due - total_hrs
    items.append(DueItem(
        category="engine", trigger="hours",
        due_at_hours=eng_due,
        due_date=_hrs_to_date(eng_rem, rate, as_of),
        hours_remaining=round(eng_rem, 1),
        days_remaining=_hrs_to_days(eng_rem, rate),
        hard_ground=eng_rem <= 0,
    ))

    # --- airframe: dual trigger (hours OR calendar, whichever first) ---
    af_interval = float(ac["airframe_inspection_interval_hours"])
    last_af_hrs = svc.get("airframe_hrs")
    last_af_date = svc.get("airframe_date")

    af_hrs_due = (last_af_hrs + af_interval) if last_af_hrs else af_interval
    af_hrs_rem = af_hrs_due - total_hrs
    af_hrs_date = _hrs_to_date(af_hrs_rem, rate, as_of)
    af_hrs_days = _hrs_to_days(af_hrs_rem, rate)

    af_cal_date = (last_af_date + timedelta(days=AIRFRAME_CALENDAR_DAYS)) if last_af_date else None
    af_cal_days = (af_cal_date - as_of).days if af_cal_date else None

    # pick whichever comes first
    if af_cal_days is not None and (af_hrs_days is None or af_cal_days < af_hrs_days):
        items.append(DueItem(
            category="airframe", trigger="dual",
            due_at_hours=af_hrs_due,
            due_date=af_cal_date,
            hours_remaining=round(af_hrs_rem, 1),
            days_remaining=af_cal_days,
            hard_ground=af_cal_days <= 0,
        ))
    else:
        items.append(DueItem(
            category="airframe", trigger="dual",
            due_at_hours=af_hrs_due,
            due_date=af_hrs_date,
            hours_remaining=round(af_hrs_rem, 1),
            days_remaining=af_hrs_days,
            hard_ground=af_hrs_rem <= 0,
        ))

    # --- avionics: calendar-based ---
    av_interval = int(ac["avionics_check_interval_days"])
    last_av = svc.get("avionics_date")
    av_due = (last_av + timedelta(days=av_interval)) if last_av else as_of + timedelta(days=av_interval)
    av_days = (av_due - as_of).days
    items.append(DueItem(
        category="avionics", trigger="calendar",
        due_at_hours=None, due_date=av_due,
        hours_remaining=None, days_remaining=av_days,
        hard_ground=av_days <= 0,
    ))

    # --- life limit (rotary) ---
    life_limit = LIFE_LIMITS.get(ac["aircraft_type"])
    if life_limit:
        life_rem = life_limit - total_hrs
        items.append(DueItem(
            category="airframe_life", trigger="life_limit",
            due_at_hours=float(life_limit),
            due_date=_hrs_to_date(life_rem, rate, as_of),
            hours_remaining=round(life_rem, 1),
            days_remaining=_hrs_to_days(life_rem, rate),
            hard_ground=life_rem <= 0,
        ))

    # sort by days remaining (None = unknown, push to end)
    items.sort(key=lambda x: x.days_remaining if x.days_remaining is not None else 999_999)
    next_due = items[0]

    # consolidation: other items due within window of soonest
    consol = []
    if next_due.days_remaining is not None:
        for item in items[1:]:
            if item.days_remaining is not None:
                gap = item.days_remaining - next_due.days_remaining
                if 0 <= gap <= CONSOLIDATION_WINDOW_DAYS:
                    consol.append(item)

    tempo_flag = (rate_30d > rate_90d * TEMPO_THRESHOLD) if rate_90d > 0 else False

    return AircraftForecast(
        aircraft_id=aid, tail_number=ac["tail_number"],
        aircraft_type=ac["aircraft_type"], total_hours=total_hrs,
        rate_30d=round(rate_30d, 2), rate_90d=round(rate_90d, 2),
        tempo_flag=tempo_flag, next_due=next_due, consolidation=consol,
    )


# -- public API --------------------------------------------------------------

def forecast(conn=None) -> list[AircraftForecast]:
    """Run maintenance forecast for all active aircraft."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    cur = conn.cursor()
    try:
        as_of = _get_as_of(cur)
        aircraft = _get_aircraft(cur)
        rates = _get_flight_rates(cur, as_of)
        last_svc = _get_last_service(cur)
        return [_forecast_one(ac, rates, last_svc, as_of) for ac in aircraft]
    finally:
        cur.close()
        if own_conn:
            conn.close()


# -- CLI output --------------------------------------------------------------

def print_report(results: list[AircraftForecast]):
    # hard grounds
    grounds = [f for f in results if f.next_due.hard_ground]
    if grounds:
        print(f"\n{'='*70}")
        print(f"HARD GROUNDS ({len(grounds)})")
        print(f"{'='*70}")
        for f in grounds:
            d = f.next_due
            detail = f"{d.hours_remaining:.0f} hrs" if d.hours_remaining is not None else f"{d.days_remaining}d"
            print(f"  {f.tail_number:<10} {f.aircraft_type:<8}  "
                  f"{d.category} [{d.trigger}]  {detail}")

    # upcoming sorted by urgency
    upcoming = sorted(
        [f for f in results if not f.next_due.hard_ground],
        key=lambda f: f.next_due.days_remaining if f.next_due.days_remaining is not None else 999_999,
    )
    print(f"\n{'='*70}")
    print(f"UPCOMING ({len(upcoming)} aircraft)")
    print(f"{'='*70}")
    hdr = (f"  {'Tail':<10} {'Type':<8} {'Item':<14} {'Trigger':<8} "
           f"{'Days':>5} {'Hrs Rem':>8} {'Due Date':>11} "
           f"{'30d':>5} {'90d':>5} {'Tempo':>5} {'Con':>3}")
    print(hdr)
    print(f"  {'-'*95}")
    for f in upcoming:
        d = f.next_due
        days = f"{d.days_remaining:>5}" if d.days_remaining is not None else "    -"
        hrs = f"{d.hours_remaining:>8.0f}" if d.hours_remaining is not None else "       -"
        ddate = d.due_date.isoformat() if d.due_date else "          -"
        tempo = "  >>>" if f.tempo_flag else ""
        con = f"{len(f.consolidation):>3}" if f.consolidation else ""
        print(f"  {f.tail_number:<10} {f.aircraft_type:<8} {d.category:<14} "
              f"{d.trigger:<8} {days} {hrs} {ddate:>11} "
              f"{f.rate_30d:>5.1f} {f.rate_90d:>5.1f} {tempo:>5} {con:>3}")

    # tempo alerts
    tempo = [f for f in results if f.tempo_flag]
    if tempo:
        print(f"\n{'='*70}")
        print(f"TEMPO ALERTS ({len(tempo)})")
        print(f"{'='*70}")
        for f in sorted(tempo, key=lambda x: x.rate_30d / max(x.rate_90d, 0.01), reverse=True):
            ratio = f.rate_30d / f.rate_90d
            print(f"  {f.tail_number:<10} {f.aircraft_type:<8}  "
                  f"30d={f.rate_30d:.1f} vs 90d={f.rate_90d:.1f} hrs/day ({ratio:.1f}x)")

    # consolidation opportunities
    consol = [f for f in results if f.consolidation]
    if consol:
        print(f"\n{'='*70}")
        print(f"CONSOLIDATION OPPORTUNITIES ({len(consol)})")
        print(f"{'='*70}")
        for f in consol:
            items = [f.next_due.category] + [c.category for c in f.consolidation]
            print(f"  {f.tail_number:<10}  {' + '.join(items)}")


if __name__ == "__main__":
    results = forecast()
    print_report(results)
