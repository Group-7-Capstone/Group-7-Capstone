"""One-off: move raw tables out of an old local DuckDB file into Parquet.

Older pulls stored every raw table inside `nyc_open_data.duckdb`. This
exports each table that has a `_provenance` row to Parquet, drops tables
with no provenance (derived copies such as `tickets_all`, which should
be views), and rewrites the DuckDB file so the freed space is returned
to disk. Run with `uv run python -m data.migrate`.
"""

import os

import duckdb

from data import base
from data.base import PROVENANCE_TABLE, parquet_path


def migrate() -> None:
    db_path = base.DB_PATH  # read at call time so tests can point it elsewhere
    if not db_path.exists():
        print(f"No database at {db_path}; nothing to migrate.")
        return

    con = duckdb.connect(str(db_path))
    try:
        tables = [
            name
            for (name,) in con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_type = 'BASE TABLE' AND table_name != ?",
                [PROVENANCE_TABLE],
            ).fetchall()
        ]
        pulled = {
            name
            for (name,) in con.execute(
                f"SELECT DISTINCT table_name FROM {PROVENANCE_TABLE}"
            ).fetchall()
        }
        for table in tables:
            if table in pulled:
                con.execute(
                    f"COPY {table} TO '{parquet_path(table).as_posix()}' "
                    "(FORMAT parquet, COMPRESSION zstd)"
                )
                expected = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                written = con.execute(
                    "SELECT COUNT(*) FROM read_parquet(?)",
                    [parquet_path(table).as_posix()],
                ).fetchone()[0]
                if written != expected:
                    raise RuntimeError(
                        f"{table}: wrote {written} rows to Parquet, expected "
                        f"{expected}; database left unchanged"
                    )
                print(f"exported {table} ({written} rows)")
            else:
                print(f"dropped {table} (no provenance; derived table)")

        compact_path = db_path.with_name(db_path.stem + "_compact.duckdb")
        compact_path.unlink(missing_ok=True)
        con.execute(f"ATTACH '{compact_path.as_posix()}' AS compact")
        con.execute(
            f"CREATE TABLE compact.{PROVENANCE_TABLE} AS "
            f"SELECT * FROM {PROVENANCE_TABLE}"
        )
        con.execute("DETACH compact")
    finally:
        con.close()

    os.replace(compact_path, db_path)
    print(f"rewrote {db_path.name} with only {PROVENANCE_TABLE}")


if __name__ == "__main__":
    migrate()
