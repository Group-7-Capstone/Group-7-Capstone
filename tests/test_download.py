"""Tests for data/download.py's SODA pagination."""

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
        [{"summons_number": "1", ":id": "row-a"}, {"summons_number": "2", ":id": "row-b"}],
        [{"summons_number": "3", ":id": "row-c"}],
    ]
    client = _FakeClient(pages)

    records = _fetch_pages(
        client, "abcd-1234", "violation_code IN (7, 36)", "tickets_test", limit=2
    )

    assert records == [
        {"summons_number": "1"},
        {"summons_number": "2"},
        {"summons_number": "3"},
    ]
    assert client.calls[0]["where"] == "violation_code IN (7, 36)"
    assert client.calls[1]["where"] == "(violation_code IN (7, 36)) AND :id > 'row-b'"


def test_fetch_pages_empty_result():
    client = _FakeClient([[]])

    records = _fetch_pages(client, "abcd-1234", "violation_code IN (7, 36)", "tickets_test")

    assert records == []
