"""Load generated CSVs into Snowflake.

Steps:
  1. Run schema.sql to create/replace tables
  2. PUT + COPY INTO for all tables
  3. Verify row counts

Usage:
    python -m data.load_to_snowflake
"""

from pathlib import Path

from utils.snowflake import get_connection

DATA_DIR = Path(__file__).parent
OUTPUT_DIR = DATA_DIR / "output"
SCHEMA_FILE = DATA_DIR / "schema.sql"

# table name -> csv filename
# order matters: dimensions before facts (foreign keys)
TABLES = [
    "bases",
    "aircraft",
    "crew",
    "missions",
    "flight_logs",
    "maintenance_events",
    "weather_observations",
]


def run_schema(cur):
    """Execute schema.sql to create/replace all tables."""
    sql = SCHEMA_FILE.read_text()
    for stmt in sql.split(";"):
        # strip comments and whitespace
        lines = [l for l in stmt.strip().splitlines() if l.strip() and not l.strip().startswith("--")]
        stmt_clean = "\n".join(lines).strip()
        if not stmt_clean:
            continue
        try:
            cur.execute(stmt_clean)
        except Exception as e:
            print(f"  warning: {e}")
    print("schema created")


def load_table(cur, table):
    """PUT csv to stage, COPY INTO table."""
    csv_path = OUTPUT_DIR / f"{table}.csv"
    if not csv_path.exists():
        print(f"  {table}: SKIP (no csv)")
        return

    stage = f"@%{table}"

    # clear any prior staged files
    cur.execute(f"REMOVE {stage}")

    # PUT local file to table stage
    # forward slashes required even on Windows
    put_path = str(csv_path.resolve()).replace("\\", "/")
    cur.execute(f"PUT 'file://{put_path}' {stage} AUTO_COMPRESS=TRUE OVERWRITE=TRUE")

    # COPY INTO — skip header, handle nulls
    copy_sql = f"""
        COPY INTO {table}
        FROM {stage}
        FILE_FORMAT = (
            TYPE = CSV
            SKIP_HEADER = 1
            FIELD_OPTIONALLY_ENCLOSED_BY = '"'
            NULL_IF = ('')
            EMPTY_FIELD_AS_NULL = TRUE
        )
        ON_ERROR = ABORT_STATEMENT
    """

    # crew.qualifications needs post-load transform (Python list repr -> JSON array)
    # load into a temp column first, then fix
    if table == "crew":
        copy_sql = f"""
            COPY INTO {table}
            FROM {stage}
            FILE_FORMAT = (
                TYPE = CSV
                SKIP_HEADER = 1
                FIELD_OPTIONALLY_ENCLOSED_BY = '"'
                NULL_IF = ('')
                EMPTY_FIELD_AS_NULL = TRUE
            )
            ON_ERROR = ABORT_STATEMENT
        """

    cur.execute(copy_sql)

    # fix qualifications: Python repr ['Bell 407', 'EC135'] -> valid JSON array
    if table == "crew":
        cur.execute("""
            UPDATE crew
            SET qualifications = PARSE_JSON(
                REPLACE(REPLACE(TO_VARCHAR(qualifications), '''', '"'), 'None', 'null')
            )
            WHERE qualifications IS NOT NULL
        """)

    rows = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"  {table}: {rows:,} rows loaded")


def verify(cur):
    """Print row counts for all tables."""
    print("\nverification:")
    for table in TABLES:
        rows = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {rows:,}")


def main():
    conn = get_connection()
    cur = conn.cursor()
    try:
        run_schema(cur)
        print("\nloading data:")
        for table in TABLES:
            load_table(cur, table)
        verify(cur)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
