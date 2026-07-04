"""FleetOps AI - decision packages front end.

Pure read/display layer over the orchestrator's output. It READS the
decision_packages Snowflake table and WRITES feedback back to it. It does
NOT run agents, the graph, or the orchestrator — presentation is decoupled
from reasoning. Reuses the existing key-pair connection in utils.snowflake.

Run:
    streamlit run app/streamlit_app.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# allow `streamlit run app/streamlit_app.py` from the repo root to import utils/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from config.settings import settings
from utils.snowflake import get_connection

# committed fallback so the cloud deploy renders without Snowflake creds
SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "data" / "decision_packages_snapshot.json"


def _live_snowflake():
    """True only when we can actually reach Snowflake (local dev). On Streamlit
    Cloud there's no key file, so we fall back to the committed snapshot."""
    try:
        return bool(settings.snowflake_account) and \
            Path(settings.snowflake_private_key_path).expanduser().is_file()
    except Exception:
        return False


SNAPSHOT_MODE = not _live_snowflake()

# vocabulary matches the orchestrator (action names, pillar labels)
PILLAR_COLORS = {
    "maintenance": "#b45309",
    "weather": "#1d4ed8",
    "demand": "#15803d",
    "staffing": "#7c3aed",
}
SIGNAL_LABELS = {
    "maintenance": "Maintenance", "weather": "Weather", "demand": "Demand",
    "staffing": "Staffing", "logistics": "Logistics",
}

LOAD_SQL = """
    SELECT package_id, region, action, priority_rank, horizon_days, confidence,
           pillars_addressed, supporting_signals, rationale, logistics_cost_nm,
           generated_at, feedback
    FROM decision_packages
    WHERE generated_at >= (
        SELECT DATEADD(minute, -5, MAX(generated_at)) FROM decision_packages
    )
    ORDER BY region, priority_rank
"""


# -- data layer --------------------------------------------------------------

def _loads(v):
    """VARIANT columns come back as JSON strings (or None)."""
    if v is None:
        return None
    return json.loads(v) if isinstance(v, str) else v


def _load_snapshot():
    """Committed JSON snapshot. VARIANT columns are already parsed; generated_at
    is an ISO string we hydrate back to datetime so the UI formats it the same."""
    rows = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    for r in rows:
        r["generated_at"] = datetime.fromisoformat(r["generated_at"])
    return rows


@st.cache_data(ttl=300)
def load_packages():
    """Latest synthesis run only (rows within 5 min of the max generated_at)."""
    if SNAPSHOT_MODE:
        return _load_snapshot()
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(LOAD_SQL)
        cols = [d[0].lower() for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        conn.close()
    for r in rows:
        r["pillars_addressed"] = _loads(r["pillars_addressed"]) or []
        r["supporting_signals"] = _loads(r["supporting_signals"]) or {}
        r["feedback"] = _loads(r["feedback"])
    return rows


def write_feedback(package_id, rating, note):
    """Capture-only feedback write-back. Logs the rating to the row; does NOT
    retrain or feed any learning loop. This is the seam a Sigma write-back
    action would fill in production."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        payload = json.dumps({"rating": rating, "note": note or None})
        cur.execute(
            "UPDATE decision_packages SET feedback = PARSE_JSON(%s) WHERE package_id = %s",
            (payload, package_id),
        )
        conn.commit()
    finally:
        conn.close()


# -- presentation helpers ----------------------------------------------------

def humanize(action):
    return action.replace("_", " ").capitalize()


def conf_label(c):
    if c >= 0.75:
        return "High"
    if c >= 0.5:
        return "Moderate"
    if c >= 0.25:
        return "Low"
    return "Very low"


def pillar_chips(pillars):
    return "".join(
        f'<span style="background:{PILLAR_COLORS.get(p, "#555")};color:white;'
        f'padding:2px 9px;border-radius:11px;font-size:0.72rem;margin-right:5px;">{p}</span>'
        for p in pillars
    )


def has_conflict(pkgs):
    """A region carries the headline tension when one action addresses both
    demand and staffing (surge demand vs constrained crew coverage)."""
    return any(
        "demand" in p["pillars_addressed"] and "staffing" in p["pillars_addressed"]
        for p in pkgs
    )


# -- rendering ---------------------------------------------------------------

def _render_feedback(pkg):
    pid = pkg["package_id"]
    current = pkg.get("feedback") or {}
    options = ["thumbs up", "thumbs down"]
    default = 0 if current.get("rating") == "up" else 1 if current.get("rating") == "down" else None

    if current.get("rating"):
        st.caption(f"Current feedback: {current['rating']}"
                   + (f" - \"{current['note']}\"" if current.get("note") else ""))

    # read-only in snapshot mode: no Snowflake to write back to on the cloud deploy
    rating = st.radio("Your rating", options, index=default, horizontal=True,
                      key=f"rate_{pid}", disabled=SNAPSHOT_MODE)
    note = st.text_area("Note (optional)", value=current.get("note") or "",
                        key=f"note_{pid}", height=68, disabled=SNAPSHOT_MODE)
    if st.button("Save feedback", key=f"save_{pid}", disabled=SNAPSHOT_MODE,
                 help="Read-only demo - captured feedback shown, write-back needs live Snowflake"
                 if SNAPSHOT_MODE else None):
        if rating is None:
            st.warning("Pick thumbs up or thumbs down first.")
        else:
            write_feedback(pid, "up" if rating == "thumbs up" else "down", note)
            load_packages.clear()
            st.success("Feedback saved.")
            st.rerun()


def _render_package(pkg):
    with st.container(border=True):
        c1, c2, c3 = st.columns([0.4, 4.0, 1.8])
        c1.markdown(f"### {pkg['priority_rank']}")
        with c2:
            st.markdown(f"**{humanize(pkg['action'])}**")
            st.markdown(pillar_chips(pkg["pillars_addressed"]), unsafe_allow_html=True)
            summary = "  |  ".join(f"{SIGNAL_LABELS.get(k, k)}: {v}"
                                   for k, v in pkg["supporting_signals"].items())
            st.caption(summary)
        with c3:
            conf = float(pkg["confidence"])
            st.progress(min(max(conf, 0.0), 1.0),
                        text=f"Confidence {conf:.2f} ({conf_label(conf)})")
            meta = f"Horizon {pkg['horizon_days']}d"
            if pkg["logistics_cost_nm"]:
                meta += f"  |  ~{pkg['logistics_cost_nm']:.0f} nm"
            st.caption(meta)

        with st.expander("Why this, and feedback"):
            st.markdown("**Rationale**")
            st.write(pkg["rationale"] or "_(no rationale recorded)_")
            st.markdown("**Supporting signals** (traced to the agents' output)")
            for k, v in pkg["supporting_signals"].items():
                st.markdown(f"- **{SIGNAL_LABELS.get(k, k.title())}:** {v}")
            st.divider()
            _render_feedback(pkg)


def render():
    st.set_page_config(page_title="FleetOps AI", layout="wide")
    packages = load_packages()

    st.title("FleetOps AI - Decision Packages")

    if not packages:
        st.info("No decision packages found. Run the orchestrator to populate "
                "decision_packages, then refresh.")
        return

    regions = sorted({p["region"] for p in packages})
    latest = max(p["generated_at"] for p in packages)
    conflict_regions = sorted({p["region"] for p in packages
                               if has_conflict([x for x in packages if x["region"] == p["region"]])})

    # sidebar: filters + freshness + seams
    st.sidebar.title("FleetOps AI")
    picked = st.sidebar.multiselect("Regions", regions, default=regions)
    sort_by = st.sidebar.radio("Sort actions by", ["Priority", "Confidence"])
    st.sidebar.caption(f"Run generated: {latest:%Y-%m-%d %H:%M}")
    st.sidebar.divider()
    if st.sidebar.button("Refresh data"):
        load_packages.clear()
        st.rerun()
    # Seam only - the app is read-only over decision_packages. A scheduled job
    # or Sigma action triggers synthesis in production; not wired here.
    st.sidebar.button("Run synthesis now", disabled=True,
                      help="Not wired. The app reads decision_packages and writes "
                           "feedback; it does not trigger the orchestrator.")

    # landing summary
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Decision packages", len(packages))
    m2.metric("Regions", len(regions))
    m3.metric("Regions in conflict", len(conflict_regions))
    m4.metric("Top-priority actions", sum(1 for p in packages if p["priority_rank"] == 1))

    if conflict_regions:
        st.warning("Conflict - surge demand vs constrained crew coverage in: "
                   + ", ".join(conflict_regions))

    st.divider()

    for region in regions:
        if region not in picked:
            continue
        region_pkgs = [p for p in packages if p["region"] == region]
        if sort_by == "Confidence":
            region_pkgs = sorted(region_pkgs, key=lambda p: -float(p["confidence"]))
        else:
            region_pkgs = sorted(region_pkgs, key=lambda p: p["priority_rank"])

        header = f"{region}  -  {len(region_pkgs)} ranked actions"
        st.subheader(header)
        if has_conflict(region_pkgs):
            top = next((p for p in region_pkgs
                        if "demand" in p["pillars_addressed"]
                        and "staffing" in p["pillars_addressed"]), None)
            if top:
                st.error("Conflict: "
                         + "  |  ".join(f"{SIGNAL_LABELS.get(k, k)}: {v}"
                                        for k, v in top["supporting_signals"].items()
                                        if k in ("demand", "staffing")))
        for pkg in region_pkgs:
            _render_package(pkg)
        st.divider()


if __name__ == "__main__":
    render()
