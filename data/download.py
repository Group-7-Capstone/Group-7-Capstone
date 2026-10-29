"""Pull raw data from NYC Open Data into the local cache.
Each pull is stored as a Parquet file (see `data/base.py`). Receipt of
each pull is logged in the `_provenance` table with the dataset ID, the
exact query, and the pull timestamp. A pull already in the cache is
served from there instead of hitting the API again, unless `force=True`.
"""

import json
import os
import shutil
import hashlib
from pathlib import Path
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


def checksum(path):
    """Hash a file without loading it into memory."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path, value):
    """Atomically replace a local receipt."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def _partition_counts(client, dataset_id, where, field):
    rows = client.get(dataset_id, select=f"date_extract_y({field}) as year,count(*) as n",
                      where=where, group=f"date_extract_y({field})", limit=100)
    return {str(row.get("year", "undated")): int(row["n"]) for row in rows}


def verify_snapshot(folder, manifest=None):
    """Check all legacy/current page receipts offline without changing the snapshot."""
    folder = Path(folder)
    manifest = manifest or json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    actual = {}
    for year, expected in sorted(manifest["expected"].items()):
        total, cursor = 0, None
        pages = sorted((folder / year).glob("*.parquet"))
        for number, page in enumerate(pages):
            if page.stem != f"{number:05d}":
                raise ValueError(f"Noncontiguous snapshot pages: {page}")
            receipt = json.loads(page.with_suffix(".json").read_text(encoding="utf-8"))
            if receipt.get("dataset_id") != manifest["dataset_id"]:
                raise ValueError(f"Page dataset mismatch: {page}")
            if manifest.get("fiscal_year") is not None and receipt.get("fiscal_year") != manifest["fiscal_year"]:
                raise ValueError(f"Page fiscal year mismatch: {page}")
            if receipt["sha256"] != checksum(page):
                raise ValueError(f"Checksum mismatch: {page}")
            if receipt.get("previous_cursor") != cursor:
                raise ValueError(f"Broken pagination checkpoint: {page}")
            parquet = pq.ParquetFile(page)
            if parquet.metadata.num_rows != receipt["rows"] or receipt["rows"] <= 0:
                raise ValueError(f"Page row-count mismatch: {page}")
            last = parquet.read_row_group(parquet.num_row_groups - 1, columns=["source_row_id"])["source_row_id"][-1].as_py()
            if last != receipt["cursor"] or last == cursor:
                raise ValueError(f"Page cursor mismatch: {page}")
            total += receipt["rows"]
            cursor = receipt["cursor"]
        if total > expected:
            raise ValueError(f"Snapshot partition exceeds expected count: {year}")
        actual[year] = total
    extra = {p.parent.name for p in folder.glob("*/*.parquet")} - set(manifest["expected"])
    if extra:
        raise ValueError(f"Unexpected snapshot partitions: {sorted(extra)}")
    return actual


def _fetch_snapshot(dataset_id, table_name, where, select, columns, folder,
                    field, fiscal_year, page_size, offline):
    """Resumable year partitions sharing the same client and pager as flat pulls."""
    manifest_path = folder / "manifest.json"
    now = lambda: datetime.now(timezone.utc).isoformat()
    client = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest["dataset_id"], manifest["where"], manifest["select"], manifest.get("fiscal_year")) != (dataset_id, where, select, fiscal_year):
            raise ValueError("Snapshot parameters differ; use a new cache")
        if manifest.get("partition_field", "issue_date") != field:
            raise ValueError("Snapshot partition field differs; use a new cache")
        actual = verify_snapshot(folder, manifest)
        if manifest.get("complete"):
            if actual != manifest["expected"] or actual != manifest.get("actual") or actual != manifest.get("final_counts"):
                raise ValueError("Snapshot totals disagree with complete receipt")
            if manifest.get("source_changed_during_pull"):
                raise ValueError("Source changed during snapshot extraction")
            return sum(actual.values())
    elif offline:
        raise FileNotFoundError(f"Missing snapshot receipt: {manifest_path}")
    else:
        folder.mkdir(parents=True, exist_ok=True)
        client = _client()
        metadata = client.get_metadata(dataset_id)
        if fiscal_year is not None and str(fiscal_year) not in metadata["name"]:
            raise ValueError("Fiscal-year dataset identity mismatch")
        save_json(folder / "metadata.json", metadata)
        manifest = {"dataset_id": dataset_id, "fiscal_year": fiscal_year, "name": metadata["name"], "partition_field": field,
                    "where": where, "select": select, "started_at": now(), "complete": False,
                    "source_rows_updated_at": metadata.get("rowsUpdatedAt"),
                    "expected": _partition_counts(client, dataset_id, where, field)}
        save_json(manifest_path, manifest)
        actual = {year: 0 for year in manifest["expected"]}
    if offline:
        raise ValueError("Snapshot is incomplete; offline reading cannot assert complete coverage")
    client = client or _client()
    try:
        for year, expected in sorted(manifest["expected"].items()):
            partition = folder / year
            partition.mkdir(exist_ok=True)
            pages = sorted(partition.glob("*.parquet"))
            cursor = json.loads(pages[-1].with_suffix(".json").read_text(encoding="utf-8"))["cursor"] if pages else None
            total = actual[year]
            period = f"{field} IS NULL" if year == "undated" else f"{field} >= '{year}-01-01' AND {field} < '{int(year)+1}-01-01'"
            if total == expected:
                continue
            query = f"({where}) AND ({period})"
            for number, rows in enumerate(_fetch_pages(client, dataset_id, query, table_name, page_size,
                                                       select=select, start_cursor=cursor), start=len(pages)):
                if total + len(rows) > expected:
                    raise ValueError("Source count increased during extraction")
                target = partition / f"{number:05d}.parquet"
                table = _page_table(rows, list(dict.fromkeys(["source_row_id", *columns])))
                tmp = target.with_suffix(".tmp")
                pq.write_table(table, tmp, compression="zstd")
                tmp.replace(target)
                receipt = {"dataset_id": dataset_id, "fiscal_year": fiscal_year,
                           "url": f"https://{SOCRATA_DOMAIN}/resource/{dataset_id}.json",
                           "query": {"select": select, "where": query if cursor is None else f"({query}) AND :id > '{cursor}'", "order": ":id", "limit": page_size},
                           "pulled_at": now(), "rows": len(rows), "previous_cursor": cursor,
                           "cursor": rows[-1]["source_row_id"], "sha256": checksum(target)}
                save_json(target.with_suffix(".json"), receipt)
                total += len(rows)
                cursor = receipt["cursor"]
                if total == expected:
                    break
            if total != expected:
                raise ValueError("Empty page before expected source count")
            actual[year] = total
        current = _partition_counts(client, dataset_id, where, field)
        metadata = client.get_metadata(dataset_id)
        changed = metadata.get("rowsUpdatedAt") != manifest.get("source_rows_updated_at")
        manifest.update(actual=actual, final_counts=current, finished_at=now(),
                        final_rows_updated_at=metadata.get("rowsUpdatedAt"),
                        source_changed_during_pull=changed,
                        complete=actual == manifest["expected"] == current and not changed)
        save_json(manifest_path, manifest)
        if not manifest["complete"]:
            raise ValueError("Source changed during snapshot extraction")
        return sum(actual.values())
    finally:
        client.close()


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
    client: Socrata, dataset_id: str, where: str, table_name: str, limit: int = 50000,
    *, select: str = "*", start_cursor: str | None = None,
) -> Iterator[list[dict]]:
    """Yield pages of a SODA query, paging by `:id` rather than `$offset`.

    Socrata's offset pagination re-scans and discards every prior row on each
    page, so cost grows with the offset; for the multi-million-row pulls here
    that turns a few minutes into hours. Keyset paging on `:id` stays O(n).
    """
    cursor = start_cursor
    projected = select != "*"
    selection = select if projected else "*,:id"
    cursor_field = "source_row_id" if projected else ":id"
    with tqdm(desc=f"{table_name} rows", unit=" rows") as pbar:
        while True:
            page_where = where if cursor is None else f"({where}) AND :id > '{cursor}'"
            page = client.get(
                dataset_id, select=selection, where=page_where, order=":id", limit=limit
            )
            if not page:
                break
            next_cursor = page[-1][cursor_field]
            if next_cursor == cursor:
                raise ValueError("Pagination cursor did not advance")
            cursor = next_cursor
            if not projected:
                for row in page:
                    row.pop(":id", None)
            yield page
            pbar.update(len(page))
            if len(page) < limit:
                break


def _page_table(page: list[dict], columns: list[str] | None = None) -> pa.Table:
    """Build an all-string Arrow table with every column seen in the page.

    SODA omits a row's null fields, so columns come from the union of all
    rows' keys, not just the first row. Values stay as SODA's strings;
    nested values (e.g. points) are JSON-encoded.
    """
    columns = columns or list(dict.fromkeys(k for row in page for k in row))
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
    dataset_id: str, table_name: str, where: str, force: bool = False,
    *, columns: list[str] | None = None, snapshot_dir: Path | None = None,
    partition_field: str | None = None, fiscal_year: int | None = None,
    page_size: int = 50000, offline: bool = False,
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
    select = "*" if columns is None else ":id as source_row_id," + ",".join(c for c in columns if c != "source_row_id")
    if snapshot_dir is not None:
        if columns is None or partition_field is None:
            raise ValueError("Partitioned snapshots require columns and partition_field")
        if force and (Path(snapshot_dir) / "manifest.json").exists():
            raise ValueError("Frozen snapshots are immutable; use a new snapshot directory")
        return _fetch_snapshot(dataset_id, table_name, where, select, columns,
                               Path(snapshot_dir), partition_field, fiscal_year, page_size, offline)
    if partition_field is not None or fiscal_year is not None:
        raise ValueError("Partition options require snapshot_dir")
    query = where if columns is None else json.dumps({"where": where, "select": select}, sort_keys=True)
    if not force:
        cached = _cached_pull(table_name, dataset_id, query)
        if cached is not None:
            return cached
    if offline:
        raise FileNotFoundError("No matching complete local cache")

    parts_dir = parquet_path(table_name).with_suffix(".parts")
    shutil.rmtree(parts_dir, ignore_errors=True)
    parts_dir.mkdir()
    try:
        n_parts = 0
        pages = (_fetch_pages(_client(), dataset_id, where, table_name) if columns is None
                 else _fetch_pages(_client(), dataset_id, where, table_name, page_size, select=select))
        for page in pages:
            pq.write_table(
                _page_table(page, None if columns is None else list(dict.fromkeys(["source_row_id", *columns]))),
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
                [table_name, dataset_id, query, datetime.now(timezone.utc)],
            )
            return con.execute(
                "SELECT COUNT(*) FROM read_parquet(?)",
                [parquet_path(table_name).as_posix()],
            ).fetchone()[0]
        finally:
            con.close()
    finally:
        shutil.rmtree(parts_dir, ignore_errors=True)
