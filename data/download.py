"""Pull raw data from NYC Open Data into the local cache.
Each pull is stored as a Parquet file (see `data/base.py`). Receipt of
each pull is logged in the `_provenance` table with the dataset ID, the
exact query, and the pull timestamp. A pull already in the cache is
served from there instead of hitting the API again, unless `force=True`.
"""

import json
import os
import shutil
from collections.abc import Iterator
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from sodapy import Socrata
from tqdm import tqdm
from urllib3.util.retry import Retry
from data.base import PROVENANCE_TABLE, get_connection, parquet_path

load_dotenv()

SOCRATA_DOMAIN = "data.cityofnewyork.us"


def _client() -> Socrata:
    token = os.getenv("SOCRATA_APP_TOKEN")
    retry = Retry(
        total=5,
        read=5,
        connect=5,
        backoff_factor=2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    adapter = {"prefix": "https://", "adapter": HTTPAdapter(max_retries=retry)}
    return Socrata(SOCRATA_DOMAIN, token, session_adapter=adapter, timeout=90)


def _cached_pull(table_name: str, dataset_id: str, where: str) -> int | None:
    """Return the cached row count, or None if this exact pull isn't cached."""
    path = parquet_path(table_name)
    if not path.exists():
        return None
    con = get_connection()
    try:
        hit = con.execute(
            f"SELECT 1 FROM {PROVENANCE_TABLE} "
            "WHERE table_name = ? AND dataset_id = ? AND query = ?",
            [table_name, dataset_id, where],
        ).fetchone()
        if hit is None:
            return None
        return con.execute(
            "SELECT COUNT(*) FROM read_parquet(?)", [path.as_posix()]
        ).fetchone()[0]
    finally:
        con.close()


def _fetch_pages(
    client: Socrata, dataset_id: str, where: str, table_name: str, limit: int = 50000
) -> Iterator[list[dict]]:
    """Yield pages of a SODA query, paging by `:id` rather than `$offset`.

    Socrata's offset pagination re-scans and discards every prior row on each
    page, so cost grows with the offset; for the multi-million-row pulls here
    that turns a few minutes into hours. Keyset paging on `:id` stays O(n).
    """
    cursor: str | None = None
    with tqdm(desc=f"{table_name} rows", unit=" rows") as pbar:
        while True:
            page_where = where if cursor is None else f"({where}) AND :id > '{cursor}'"
            page = client.get(
                dataset_id, select="*,:id", where=page_where, order=":id", limit=limit
            )
            if not page:
                break
            cursor = page[-1][":id"]
            for row in page:
                row.pop(":id", None)
            yield page
            pbar.update(len(page))
            if len(page) < limit:
                break


def _page_table(page: list[dict]) -> pa.Table:
    """Build an all-string Arrow table with every column seen in the page.

    SODA omits a row's null fields, so columns come from the union of all
    rows' keys, not just the first row. Values stay as SODA's strings;
    nested values (e.g. points) are JSON-encoded.
    """
    columns = list(dict.fromkeys(k for row in page for k in row))
    rows = [
        {
            k: v if v is None or isinstance(v, str) else json.dumps(v)
            for k, v in row.items()
        }
        for row in page
    ]
    return pa.Table.from_pylist(
        rows, schema=pa.schema([(c, pa.string()) for c in columns])
    )


def fetch_soda(
    dataset_id: str, table_name: str, where: str, force: bool = False
) -> int:
    """Pull a filtered SODA dataset into the local cache and return its row count.

    `where` is the explicit, logged pull parameter (e.g. a violation-code
    filter) required by AGENTS.md before pulling any large dataset.

    Each page is written straight to its own Parquet part file, so only one
    page is ever held in memory. SODA omits null fields from its JSON, so
    pages can have different columns; DuckDB merges the parts by column name
    into the table's single Parquet file. Read the data with
    `data.reader.read_table` or SQL on the table name.
    """
    if not force:
        cached = _cached_pull(table_name, dataset_id, where)
        if cached is not None:
            return cached

    parts_dir = parquet_path(table_name).with_suffix(".parts")
    shutil.rmtree(parts_dir, ignore_errors=True)
    parts_dir.mkdir()
    try:
        n_parts = 0
        for page in _fetch_pages(_client(), dataset_id, where, table_name):
            pq.write_table(
                _page_table(page),
                parts_dir / f"{n_parts:06d}.parquet",
            )
            n_parts += 1
        if n_parts == 0:
            raise ValueError(f"{dataset_id} returned no rows for: {where}")

        con = get_connection()
        try:
            con.execute(
                "COPY (SELECT * FROM read_parquet("
                f"'{(parts_dir / '*.parquet').as_posix()}', union_by_name = true)) "
                f"TO '{parquet_path(table_name).as_posix()}' "
                "(FORMAT parquet, COMPRESSION zstd)"
            )
            con.execute(
                f"DELETE FROM {PROVENANCE_TABLE} WHERE table_name = ?", [table_name]
            )
            con.execute(
                f"INSERT INTO {PROVENANCE_TABLE} VALUES (?, ?, ?, ?)",
                [table_name, dataset_id, where, datetime.now(timezone.utc)],
            )
            return con.execute(
                "SELECT COUNT(*) FROM read_parquet(?)",
                [parquet_path(table_name).as_posix()],
            ).fetchone()[0]
        finally:
            con.close()
    finally:
        shutil.rmtree(parts_dir, ignore_errors=True)
