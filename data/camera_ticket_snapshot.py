"""Resumable, projected camera-ticket snapshot; raw values remain strings."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/raw/camera_ticket_trends_20261014"
START, END = "2016-01-01", "2026-08-31"
SOURCES = dict(zip(range(2017, 2028), [
    "2bnn-yakx", "a5td-mswe", "faiq-9dfq", "p7t3-5i9s", "kvfd-bves",
    "7mxj-7a6y", "869v-vr48", "8zf9-spf8", "m5vz-tzqv", "9mwx-gamw", "pvqr-7yc4",
]))
FIELDS = ["source_row_id", "summons_number", "issue_date", "violation_code",
          "street_name", "intersecting_street", "violation_county", "issuing_agency"]
SELECT = ":id as source_row_id," + ",".join(FIELDS[1:])
BASE_WHERE = ("violation_code IN (7,36) AND "
              f"((issue_date >= '{START}' AND issue_date < '{END}') OR issue_date IS NULL)")


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def session():
    load_dotenv(ROOT / ".env")
    client = requests.Session()
    client.mount("https://", HTTPAdapter(max_retries=Retry(
        total=6, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])))
    if os.getenv("SOCRATA_APP_TOKEN"):
        client.headers["X-App-Token"] = os.environ["SOCRATA_APP_TOKEN"]
    return client


def get_json(client, url, params=None):
    response = client.get(url, params=params, timeout=(30, 180))
    response.raise_for_status()
    return response.json()


def year_counts(client, dataset):
    return get_json(client, f"https://data.cityofnewyork.us/resource/{dataset}.json", {
        "$select": "date_extract_y(issue_date) as year,count(*) as n",
        "$where": BASE_WHERE, "$group": "date_extract_y(issue_date)", "$limit": 100,
    })


def normalized_counts(rows):
    return {str(r.get("year", "undated")): int(r["n"]) for r in rows}


def fetch_source(fiscal_year, dataset, page_size=250000):
    folder = CACHE / f"fy{fiscal_year}"
    folder.mkdir(parents=True, exist_ok=True)
    client = session()
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        metadata = get_json(client, f"https://data.cityofnewyork.us/api/views/{dataset}.json")
        if str(fiscal_year) not in metadata["name"]:
            raise ValueError(f"Fiscal-year identity mismatch: {metadata['name']}")
        save_json(folder / "metadata.json", metadata)
        manifest = {"dataset_id": dataset, "fiscal_year": fiscal_year,
                    "name": metadata["name"], "where": BASE_WHERE, "select": SELECT,
                    "started_at": stamp(), "source_rows_updated_at": metadata.get("rowsUpdatedAt"),
                    "expected": normalized_counts(year_counts(client, dataset)), "complete": False}
        save_json(manifest_path, manifest)
    if manifest["where"] != BASE_WHERE or manifest["select"] != SELECT:
        raise ValueError("Snapshot parameters differ; use a new cache.")
    print(f"FY{fiscal_year}: expected {sum(manifest['expected'].values()):,}", flush=True)
    actual = {}
    for year, expected in sorted(manifest["expected"].items()):
        partition = folder / year
        partition.mkdir(exist_ok=True)
        cursor, total, page = None, 0, 0
        while total < expected:
            target = partition / f"{page:05d}.parquet"
            receipt_path = target.with_suffix(".json")
            if target.exists() and receipt_path.exists():
                receipt = json.loads(receipt_path.read_text())
                if receipt["sha256"] != checksum(target):
                    raise ValueError(f"Checksum mismatch: {target}")
                if receipt.get("previous_cursor") != cursor:
                    raise ValueError("Broken pagination checkpoint")
            else:
                period = "issue_date IS NULL" if year == "undated" else (
                    f"issue_date >= '{year}-01-01' AND issue_date < '{int(year)+1}-01-01'")
                where = f"({BASE_WHERE}) AND ({period})"
                if cursor:
                    where += f" AND :id > '{cursor}'"
                params = {"$select": SELECT, "$where": where, "$order": ":id", "$limit": page_size}
                url = f"https://data.cityofnewyork.us/resource/{dataset}.csv"
                tmp = target.with_suffix(".download")
                for attempt in range(5):
                    try:
                        with client.get(url, params=params, stream=True, timeout=(30, 180)) as response:
                            response.raise_for_status()
                            request_url = response.url
                            with tmp.open("wb") as stream:
                                for block in response.iter_content(1024 * 1024):
                                    stream.write(block)
                        table = pacsv.read_csv(tmp, convert_options=pacsv.ConvertOptions(
                            column_types={k: pa.string() for k in FIELDS}, strings_can_be_null=True))
                        if table.num_rows == 0:
                            raise ValueError("Empty page before expected source count")
                        pq.write_table(table, target, compression="zstd")
                        receipt = {"url": request_url, "dataset_id": dataset, "fiscal_year": fiscal_year,
                                   "pulled_at": stamp(), "rows": table.num_rows,
                                   "previous_cursor": cursor,
                                   "cursor": table["source_row_id"][-1].as_py(), "sha256": checksum(target)}
                        save_json(receipt_path, receipt)
                        tmp.unlink(missing_ok=True)
                        break
                    except Exception:
                        if attempt == 4:
                            raise
                        time.sleep(min(30, 2 ** attempt))
            total += receipt["rows"]
            cursor = receipt["cursor"]
            page += 1
            print(f"FY{fiscal_year}/{year}: {total:,}/{expected:,}", flush=True)
        if total != expected:
            raise ValueError(f"Source count changed in FY{fiscal_year}/{year}")
        actual[year] = total
    current = normalized_counts(year_counts(client, dataset))
    metadata = get_json(client, f"https://data.cityofnewyork.us/api/views/{dataset}.json")
    manifest.update(actual=actual, final_counts=current, finished_at=stamp(),
                    final_rows_updated_at=metadata.get("rowsUpdatedAt"),
                    complete=actual == manifest["expected"] == current)
    manifest["source_changed_during_pull"] = (
        metadata.get("rowsUpdatedAt") != manifest.get("source_rows_updated_at"))
    save_json(manifest_path, manifest)
    print(f"FY{fiscal_year}: complete={manifest['complete']}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    codes = get_json(session(), "https://data.cityofnewyork.us/resource/ncbg-6agr.json", {"$limit": 1000})
    save_json(CACHE / "official_violation_codes.json", codes)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(fetch_source, fy, ds): fy for fy, ds in SOURCES.items()}
        for task in as_completed(tasks):
            try:
                task.result()
            except Exception as error:
                failures.append({"fiscal_year": tasks[task], "error": str(error)})
                print(f"FAILED {tasks[task]}: {error}", flush=True)
    save_json(CACHE / "download_failures.json", failures)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
