"""Pull raw data from NYC Open Data into the local DuckDB cache.
Receipt of each pull is logged in the `_provenance` table with the dataset ID, the
exact query, and the pull timestamp. A pull already in the cache is
served from there instead of hitting the API again, unless `force=True`.
"""

import os
from datetime import datetime, timezone
import pandas as pd
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from sodapy import Socrata
from tqdm import tqdm
from urllib3.util.retry import Retry
from data.base import PROVENANCE_TABLE, get_connection

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


def _cached_pull(table_name: str, dataset_id: str, where: str) -> pd.DataFrame | None:
    con = get_connection()
    try:
        hit = con.execute(
            f"SELECT 1 FROM {PROVENANCE_TABLE} "
            "WHERE table_name = ? AND dataset_id = ? AND query = ?",
            [table_name, dataset_id, where],
        ).fetchone()
        if hit is None:
            return None
        return con.execute(f"SELECT * FROM {table_name}").fetchdf()
    finally:
        con.close()


def _fetch_pages(
    client: Socrata, dataset_id: str, where: str, table_name: str, limit: int = 50000
) -> list[dict]:
    """Page through a SODA query by `:id` rather than `$offset`.

    Socrata's offset pagination re-scans and discards every prior row on each
    page, so cost grows with the offset; for the multi-million-row pulls here
    that turns a few minutes into hours. Keyset paging on `:id` stays O(n).
    """
    records: list[dict] = []
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
            records.extend(page)
            pbar.update(len(page))
            if len(page) < limit:
                break
    return records


def fetch_soda(
    dataset_id: str, table_name: str, where: str, force: bool = False
) -> pd.DataFrame:
    """Pull a filtered SODA dataset into the local cache.

    `where` is the explicit, logged pull parameter (e.g. a violation-code
    filter) required by AGENTS.md before pulling any large dataset.
    """
    if not force:
        cached = _cached_pull(table_name, dataset_id, where)
        if cached is not None:
            return cached

    records = _fetch_pages(_client(), dataset_id, where, table_name)
    df = pd.DataFrame.from_records(records)

    con = get_connection()
    try:
        con.register("_df_tmp", df)
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM _df_tmp")
        con.execute(
            f"DELETE FROM {PROVENANCE_TABLE} WHERE table_name = ?", [table_name]
        )
        con.execute(
            f"INSERT INTO {PROVENANCE_TABLE} VALUES (?, ?, ?, ?)",
            [table_name, dataset_id, where, datetime.now(timezone.utc)],
        )
    finally:
        con.close()
    return df
