"""Tests for data/download.py: SODA pagination and LION releases."""

import io
import zipfile

import duckdb

from data import download
from data.download import _fetch_pages


class _FakeClient:
    """Stands in for sodapy.Socrata, serving fixed pages keyed by cursor."""

    def __init__(self, pages: list[list[dict]]):
        self.pages = pages
        self.calls: list[dict] = []

    def get(self, dataset_id, select, where, order, limit):
        self.calls.append({"where": where, "limit": limit})
        return self.pages.pop(0) if self.pages else []


def test_fetch_pages_pages_by_id_cursor_not_offset():
    pages = [
        [
            {"summons_number": "1", ":id": "row-a"},
            {"summons_number": "2", ":id": "row-b"},
        ],
        [{"summons_number": "3", ":id": "row-c"}],
    ]
    client = _FakeClient(pages)

    pages = list(
        _fetch_pages(
            client, "abcd-1234", "violation_code IN (7, 36)", "tickets_test", limit=2
        )
    )

    assert pages == [
        [{"summons_number": "1"}, {"summons_number": "2"}],
        [{"summons_number": "3"}],
    ]
    assert client.calls[0]["where"] == "violation_code IN (7, 36)"
    assert client.calls[1]["where"] == "(violation_code IN (7, 36)) AND :id > 'row-b'"


def test_fetch_pages_empty_result():
    client = _FakeClient([[]])

    pages = list(
        _fetch_pages(client, "abcd-1234", "violation_code IN (7, 36)", "tickets_test")
    )

    assert pages == []


def test_download_lion_unzips_release_and_logs_provenance(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("lion/lion.gdb/a00000001.gdbtable", b"x")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, chunk_size):
            yield buffer.getvalue()

    db = str(tmp_path / "cache.duckdb")
    with duckdb.connect(db) as con:
        con.execute(
            "CREATE TABLE _provenance (table_name VARCHAR, dataset_id VARCHAR, "
            "query VARCHAR, pulled_at TIMESTAMP)"
        )
    urls = []
    monkeypatch.setattr(download, "lion_release_dir", lambda r: tmp_path / r)
    monkeypatch.setattr(download, "get_connection", lambda: duckdb.connect(db))
    monkeypatch.setattr(
        download.requests, "get", lambda url, **kw: urls.append(url) or _Response()
    )

    download.download_lion("22c")
    download.download_lion("22c")  # already there: no second download

    assert (tmp_path / "22c" / "lion" / "lion.gdb" / "a00000001.gdbtable").exists()
    assert urls == [download.LION_URL.format(release="22c")]
    with duckdb.connect(db) as con:
        logged = con.execute("SELECT table_name, query FROM _provenance").fetchall()
    assert logged == [("lion_22c", urls[0])]
