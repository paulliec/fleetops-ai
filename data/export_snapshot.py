"""One-off: dump the latest decision_packages run to a committed JSON snapshot
so the Streamlit Cloud deploy can render without Snowflake creds.

    py data/export_snapshot.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.snowflake import get_connection

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

OUT = Path(__file__).resolve().parent / "decision_packages_snapshot.json"


def _loads(v):
    if v is None:
        return None
    return json.loads(v) if isinstance(v, str) else v


def main():
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
        # generated_at -> ISO string; confidence/logistics may be Decimal
        r["generated_at"] = r["generated_at"].isoformat()
        r["confidence"] = float(r["confidence"]) if r["confidence"] is not None else None
        r["logistics_cost_nm"] = (
            float(r["logistics_cost_nm"]) if r["logistics_cost_nm"] is not None else None
        )

    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} packages to {OUT}")


if __name__ == "__main__":
    main()
