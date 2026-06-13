"""Weather impact agent — standalone module, no LLM or LangGraph.

Pulls base coordinates from Snowflake, fetches the hourly forecast per
base from Open-Meteo (free, no key), and classifies each hour as
flyable / marginal / no_fly. Collapses consecutive same-category hours
into windows so the orchestrator gets timing, not a wall of hours.

Note: Open-Meteo doesn't report cloud ceiling, so this isn't a true
METAR flight-category (VFR/MVFR/IFR/LIFR) classifier — those need
ceiling. We classify on visibility, sustained wind, gusts, and WMO
weather codes, which is what Open-Meteo reports reliably.
TODO: fix later - fold in ceiling once a METAR source is wired in.

Usage:
    python -m agents.weather
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import requests

from utils.snowflake import get_connection

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
FORECAST_HOURS = 72
HOURLY_VARS = "temperature_2m,wind_speed_10m,wind_gusts_10m,visibility,precipitation,weather_code,cloud_cover"

# classification thresholds (statute miles, knots)
VIS_NOFLY = 1.0
VIS_MARGINAL = 3.0
WIND_MARGINAL = 25
WIND_NOFLY = 35
GUST_MARGINAL = 25
GUST_NOFLY = 40

METERS_PER_SM = 1609.34

# WMO weather codes. Freezing precip, heavy snow, and thunderstorms ground
# the fleet; lighter precip and fog are marginal (visibility carries the rest).
NOFLY_CODES = {56, 57, 66, 67, 75, 82, 86, 95, 96, 99}
MARGINAL_CODES = {45, 48, 51, 53, 55, 61, 63, 65, 71, 73, 80, 81, 85}

CATEGORY_RANK = {"flyable": 0, "marginal": 1, "no_fly": 2}


@dataclass
class WeatherWindow:
    category: str           # flyable | marginal | no_fly
    start: datetime
    end: datetime           # exclusive — window covers [start, end)
    hours: int
    factors: list[str] = field(default_factory=list)  # what drove the rating


@dataclass
class BaseForecast:
    base_id: int
    name: str
    icao_code: str
    lat: float
    lon: float
    flyable_hours: int
    marginal_hours: int
    no_fly_hours: int
    next_no_fly: WeatherWindow | None
    windows: list[WeatherWindow] = field(default_factory=list)


# -- data fetching -----------------------------------------------------------

def _get_bases(cur):
    cur.execute("SELECT base_id, name, icao_code, lat, lon FROM bases ORDER BY base_id")
    cols = [d[0].lower() for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _fetch_forecast(lat, lon):
    """Hourly forecast for the next FORECAST_HOURS, base-local time."""
    resp = requests.get(OPEN_METEO_URL, params={
        "latitude": lat,
        "longitude": lon,
        "hourly": HOURLY_VARS,
        "wind_speed_unit": "kn",
        "forecast_hours": FORECAST_HOURS,
        "timezone": "auto",
    }, timeout=20)
    resp.raise_for_status()
    h = resp.json()["hourly"]
    return [
        {
            "time": datetime.fromisoformat(h["time"][i]),
            "wind": h["wind_speed_10m"][i],
            "gust": h["wind_gusts_10m"][i],
            "vis_sm": h["visibility"][i] / METERS_PER_SM,
            "code": h["weather_code"][i],
        }
        for i in range(len(h["time"]))
    ]


# -- classification ----------------------------------------------------------

def _classify_hour(hour):
    """Return (category, limiting_factor). Worst constraint wins."""
    no_fly, marginal = [], []

    if hour["vis_sm"] < VIS_NOFLY:
        no_fly.append("visibility")
    elif hour["vis_sm"] < VIS_MARGINAL:
        marginal.append("visibility")

    if hour["wind"] >= WIND_NOFLY:
        no_fly.append("wind")
    elif hour["wind"] >= WIND_MARGINAL:
        marginal.append("wind")

    if hour["gust"] >= GUST_NOFLY:
        no_fly.append("wind_gust")
    elif hour["gust"] >= GUST_MARGINAL:
        marginal.append("wind_gust")

    if hour["code"] in NOFLY_CODES:
        no_fly.append("weather")
    elif hour["code"] in MARGINAL_CODES:
        marginal.append("weather")

    if no_fly:
        return "no_fly", no_fly[0]
    if marginal:
        return "marginal", marginal[0]
    return "flyable", None


def _build_windows(hours):
    """Collapse consecutive same-category hours into windows."""
    windows = []
    cur = None
    for hour in hours:
        cat, factor = _classify_hour(hour)
        if cur and cur.category == cat:
            cur.end = hour["time"] + timedelta(hours=1)
            cur.hours += 1
            if factor and factor not in cur.factors:
                cur.factors.append(factor)
        else:
            cur = WeatherWindow(
                category=cat,
                start=hour["time"],
                end=hour["time"] + timedelta(hours=1),
                hours=1,
                factors=[factor] if factor else [],
            )
            windows.append(cur)
    return windows


def _assess_one(base, hours):
    windows = _build_windows(hours)
    counts = {"flyable": 0, "marginal": 0, "no_fly": 0}
    for w in windows:
        counts[w.category] += w.hours
    next_no_fly = next((w for w in windows if w.category == "no_fly"), None)
    return BaseForecast(
        base_id=base["base_id"], name=base["name"], icao_code=base["icao_code"],
        lat=float(base["lat"]), lon=float(base["lon"]),
        flyable_hours=counts["flyable"], marginal_hours=counts["marginal"],
        no_fly_hours=counts["no_fly"], next_no_fly=next_no_fly, windows=windows,
    )


# -- public API --------------------------------------------------------------

def assess(conn=None) -> list[BaseForecast]:
    """Run weather assessment for all bases."""
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    cur = conn.cursor()
    try:
        bases = _get_bases(cur)
    finally:
        cur.close()
        if own_conn:
            conn.close()
    return [_assess_one(b, _fetch_forecast(b["lat"], b["lon"])) for b in bases]


def to_dict(results: list[BaseForecast]) -> list[dict]:
    """JSON-serializable view — what the orchestrator will consume."""
    return [
        {
            "base_id": f.base_id, "name": f.name, "icao_code": f.icao_code,
            "lat": f.lat, "lon": f.lon,
            "flyable_hours": f.flyable_hours, "marginal_hours": f.marginal_hours,
            "no_fly_hours": f.no_fly_hours,
            "next_no_fly": _window_dict(f.next_no_fly) if f.next_no_fly else None,
            "windows": [_window_dict(w) for w in f.windows],
        }
        for f in results
    ]


def _window_dict(w: WeatherWindow) -> dict:
    return {
        "category": w.category,
        "start": w.start.isoformat(),
        "end": w.end.isoformat(),
        "hours": w.hours,
        "factors": w.factors,
    }


# -- CLI output --------------------------------------------------------------

def print_report(results: list[BaseForecast]):
    print(f"\n{'='*70}")
    print(f"WEATHER OUTLOOK - next {FORECAST_HOURS}h, {len(results)} bases")
    print(f"{'='*70}")
    print(f"  {'Base':<28} {'ICAO':<5} {'Fly':>5} {'Marg':>5} {'NoFly':>6}  Next no-fly")
    print(f"  {'-'*68}")
    for f in sorted(results, key=lambda x: x.no_fly_hours, reverse=True):
        nf = ""
        if f.next_no_fly:
            w = f.next_no_fly
            nf = f"{w.start:%a %H:%M} ({w.hours}h, {'/'.join(w.factors)})"
        print(f"  {f.name:<28} {f.icao_code:<5} {f.flyable_hours:>5} "
              f"{f.marginal_hours:>5} {f.no_fly_hours:>6}  {nf}")

    # per-base window detail
    for f in results:
        print(f"\n  {f.name} ({f.icao_code})")
        for w in f.windows:
            factors = f"  [{'/'.join(w.factors)}]" if w.factors else ""
            print(f"    {w.start:%a %m-%d %H:%M} -> {w.end:%H:%M}  "
                  f"{w.category:<8} {w.hours}h{factors}")


if __name__ == "__main__":
    results = assess()
    print_report(results)
