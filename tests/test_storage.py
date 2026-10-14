"""Tests for Parquet-backed raw storage in data/base.py, download.py and migrate.py."""

import duckdb
import pytest

import data.base
import data.download
from data.base import PROVENANCE_TABLE, get_connection, parquet_path
from data.migrate import migrate
from data.reader import read_table

WHERE = "violation_code IN (7, 36)"


@pytest.fixture(autouse=True)
def tmp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(data.base, "DB_PATH", tmp_path / "test.duckdb")


def _fake_pull(monkeypatch, pages):
    calls = []

    def fake_fetch_pages(client, dataset_id, where, table_name):
        calls.append(dataset_id)
        for page in pages:
            yield [dict(r) for r in page]

    monkeypatch.setattr(data.download, "_client", lambda: None)
    monkeypatch.setattr(data.download, "_fetch_pages", fake_fetch_pages)
    return calls


def test_fetch_soda_writes_parquet_not_a_table(monkeypatch):
    _fake_pull(monkeypatch, [[{"summons_number": "1"}, {"summons_number": "2"}]])

    n_rows = data.download.fetch_soda("abcd-1234", "tickets_test", WHERE)

    assert n_rows == 2

    assert parquet_path("tickets_test").exists()
    con = get_connection()
    try:
        base_tables = con.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_type = 'BASE TABLE'"
        ).fetchall()
        provenance = con.execute(
            f"SELECT dataset_id, query FROM {PROVENANCE_TABLE}"
        ).fetchall()
    finally:
        con.close()
    assert base_tables == [(PROVENANCE_TABLE,)]
    assert provenance == [("abcd-1234", WHERE)]
    assert read_table("tickets_test")["summons_number"].tolist() == ["1", "2"]


def test_fetch_soda_keeps_columns_missing_from_some_rows_and_pages(monkeypatch):
    # SODA omits null fields, so columns vary by row and by page
    _fake_pull(
        monkeypatch,
        [
            [{"summons_number": "1"}, {"summons_number": "2", "street_name": "A ST"}],
            [{"summons_number": "3", "location": {"type": "Point"}}],
        ],
    )

    n_rows = data.download.fetch_soda("abcd-1234", "tickets_test", WHERE)

    df = read_table("tickets_test").sort_values("summons_number")
    assert n_rows == 3
    assert df["summons_number"].tolist() == ["1", "2", "3"]
    assert df["street_name"].isna().tolist() == [True, False, True]
    assert df["location"].tolist()[2] == '{"type": "Point"}'
    assert not parquet_path("tickets_test").with_suffix(".parts").exists()


def test_fetch_soda_serves_cache_without_refetching(monkeypatch):
    calls = _fake_pull(monkeypatch, [[{"summons_number": "1"}]])

    data.download.fetch_soda("abcd-1234", "tickets_test", WHERE)
    cached = data.download.fetch_soda("abcd-1234", "tickets_test", WHERE)

    assert calls == ["abcd-1234"]
    assert cached == 1


def test_fetch_soda_refetches_when_parquet_missing(monkeypatch):
    calls = _fake_pull(monkeypatch, [[{"summons_number": "1"}]])

    data.download.fetch_soda("abcd-1234", "tickets_test", WHERE)
    parquet_path("tickets_test").unlink()
    data.download.fetch_soda("abcd-1234", "tickets_test", WHERE)

    assert calls == ["abcd-1234", "abcd-1234"]


def test_saved_view_over_parquet_tables_works_in_new_connection(monkeypatch):
    _fake_pull(monkeypatch, [[{"summons_number": "1"}]])
    data.download.fetch_soda("abcd-1234", "tickets_a", WHERE)
    _fake_pull(monkeypatch, [[{"summons_number": "2", "extra": "x"}]])
    data.download.fetch_soda("efgh-5678", "tickets_b", WHERE)

    con = get_connection()
    con.execute(
        "CREATE OR REPLACE VIEW tickets_all AS "
        "SELECT * FROM tickets_a UNION ALL BY NAME SELECT * FROM tickets_b"
    )
    con.close()

    df = read_table("tickets_all").sort_values("summons_number")
    assert df["summons_number"].tolist() == ["1", "2"]
    assert df["extra"].isna().tolist() == [True, False]


def test_migrate_exports_pulled_tables_and_drops_derived():
    con = duckdb.connect(str(data.base.DB_PATH))
    con.execute(
        f"CREATE TABLE {PROVENANCE_TABLE} "
        "(table_name VARCHAR, dataset_id VARCHAR, query VARCHAR, pulled_at TIMESTAMP)"
    )
    con.execute(
        f"INSERT INTO {PROVENANCE_TABLE} VALUES "
        f"('tickets_a', 'abcd-1234', '{WHERE}', now())"
    )
    con.execute("CREATE TABLE tickets_a AS SELECT '1' AS summons_number")
    con.execute("CREATE TABLE tickets_all AS SELECT * FROM tickets_a")
    con.close()

    migrate()

    assert parquet_path("tickets_a").exists()
    assert not parquet_path("tickets_all").exists()
    con = duckdb.connect(str(data.base.DB_PATH))
    try:
        tables = con.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
    finally:
        con.close()
    assert tables == [(PROVENANCE_TABLE,)]
    assert read_table("tickets_a")["summons_number"].tolist() == ["1"]
