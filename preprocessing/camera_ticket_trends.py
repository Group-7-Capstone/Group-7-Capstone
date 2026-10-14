"""Ticket counts on Eric's unchanged location groups and activation dates."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from data.camera_ticket_snapshot import CACHE, START, END, SOURCES, checksum, save_json
from data.reader import iter_camera_ticket_batches, read_eric_registry, read_snapshot_inventory

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / "yihan/notebook-deliverables/camera-ticket-trends"
OUT = DELIVERY / "outputs"
WORK = ROOT / "data/processed/camera_ticket_trends"
BASE_WEEK = pd.Timestamp(START).to_period("W-SUN").start_time
CUTOFF = pd.Timestamp(END) - pd.Timedelta(days=1)
PARTITIONS = 64


def eric_address(street, cross):
    """Exact normalization from Eric's multi-year notebook, including its limitations."""
    return ((street.fillna("") + " " + cross.fillna("")).str.upper()
            .str.replace(r"\b(NB|SB|EB|WB|N/B|S/B|E/B|W/B)\b", "", regex=True)
            .str.replace(r"\(.*?\)", "", regex=True).str.replace("@", " AND ", regex=False)
            .str.replace(r"^\s+", "", regex=True).str.replace(r"\s+", " ", regex=True)
            .str.strip() + ", NEW YORK, NY")


def attach_identity(cameras):
    cameras = cameras.copy()
    cameras["camera_id"] = ["cam_" + hashlib.sha256(
        (q + "\0" + t).encode()).hexdigest()[:16] for q, t in
        zip(cameras.search_query, cameras.camera_type)]
    assert cameras.camera_id.is_unique
    assert not cameras.duplicated(["search_query", "camera_type"]).any()
    cameras["camera_index"] = np.arange(len(cameras))
    cameras["activation"] = pd.to_datetime(cameras.inferred_activation_date)
    return cameras


def deduplicate(frame):
    """Newest fiscal year wins; row ID breaks ties deterministically."""
    ordered = frame.sort_values(["summons_number", "source_fiscal_year", "source_row_id"],
                                ascending=[True, False, False], kind="stable")
    duplicates = ordered[ordered.summons_number.duplicated(keep=False)].copy()
    if len(duplicates):
        fields = ["issue_date", "violation_code", "street_name", "intersecting_street",
                  "violation_county", "issuing_agency"]
        variants = duplicates.drop_duplicates(["summons_number", *fields]).groupby("summons_number").size()
        duplicates["conflicting_values"] = duplicates.summons_number.map(variants).gt(1)
        duplicates["selected_record"] = ~duplicates.summons_number.duplicated()
    return ordered.drop_duplicates("summons_number"), duplicates


def partition_tickets(cache=CACHE):
    """Partition by ticket identifier so cross-year duplicates stay together."""
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = read_snapshot_inventory(cache)
    fingerprint = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()
    state_path = WORK / "partitions.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["fingerprint"] == fingerprint and all((WORK/f"raw_{i:02d}.parquet").exists() for i in range(PARTITIONS)):
            return state, inventory
    writers, counts = {}, Counter()
    try:
        for fy, frame in iter_camera_ticket_batches(cache):
            counts["source_records"] += len(frame)
            frame["source_fiscal_year"] = fy
            valid_id = frame.summons_number.notna() & frame.summons_number.fillna("").str.strip().ne("")
            missing = frame[~valid_id]
            if len(missing):
                missing.to_parquet(OUT/f"missing_identifiers_{fy}_{counts['source_records']}.parquet", index=False)
            counts["missing_identifier_records"] += len(missing)
            frame = frame[valid_id].copy()
            frame["summons_number"] = frame.summons_number.str.strip()
            buckets = pd.util.hash_pandas_object(frame.summons_number, index=False, categorize=False) % PARTITIONS
            for bucket, part in frame.groupby(buckets):
                bucket = int(bucket)
                table = pa.Table.from_pandas(part, preserve_index=False)
                # SODA may omit a field entirely in a small batch; enforce string schema.
                schema = pa.schema([(k, pa.int64() if k == "source_fiscal_year" else pa.string()) for k in part.columns])
                table = table.cast(schema)
                if bucket not in writers:
                    writers[bucket] = pq.ParquetWriter(WORK/f"raw_{bucket:02d}.parquet", schema, compression="zstd")
                writers[bucket].write_table(table)
            if counts["source_records"] % 1000000 < 100000:
                print(f"Partitioned {counts['source_records']:,} raw records", flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    state = {"fingerprint": fingerprint, "counts": dict(counts)}
    save_json(state_path, state)
    return state, inventory


def make_weekly(camera_id, activation, last_date, counts, coverage=None):
    """Only fill observed-snapshot zeros; retain unavailable weeks as null."""
    if pd.isna(last_date) or last_date < activation:
        return pd.DataFrame()
    start = max(pd.Timestamp(START), pd.Timestamp(activation))
    weeks = pd.date_range(start.to_period("W-SUN").start_time,
                          pd.Timestamp(last_date).to_period("W-SUN").start_time, freq="W-MON")
    offsets = ((weeks - BASE_WEEK).days // 7).astype(int)
    available = np.ones(len(weeks), dtype=bool) if coverage is None else np.asarray(coverage)[offsets]
    full = (weeks >= start) & ((weeks + pd.Timedelta(days=6)) <= pd.Timestamp(last_date)) & available
    frame = pd.DataFrame({"camera_id": camera_id, "week_start": weeks,
                          "weeks_since_activation": ((weeks - pd.Timestamp(activation)).days // 7),
                          "observed_ticket_count": np.asarray(counts)[offsets],
                          "source_complete": available, "is_complete_week": full})
    frame["ticket_count"] = frame.observed_ticket_count.astype(float).where(available)
    frame["moving_average_4w"] = frame.ticket_count.where(full).rolling(4, min_periods=4).mean()
    frame["is_partial_week"] = ~((weeks >= start) & ((weeks + pd.Timedelta(days=6)) <= pd.Timestamp(last_date)))
    return frame


def comparison_windows(weekly):
    result = dict(first_13w_mean=None, last_13w_mean=None, absolute_change=None,
                  percentage_change=None, comparison_start=None, comparison_end=None)
    if weekly.empty:
        return result
    # Choose the longest fully covered contiguous run; earliest run breaks a tie.
    runs, run = [], []
    for i, valid in enumerate(weekly.is_complete_week):
        if valid:
            run.append(i)
        elif run:
            runs.append(run); run = []
    if run:
        runs.append(run)
    eligible = max(runs, key=len, default=[])
    if len(eligible) >= 26:
        block = weekly.iloc[eligible]
        first, last = float(block.ticket_count.iloc[:13].mean()), float(block.ticket_count.iloc[-13:].mean())
        result.update(first_13w_mean=first, last_13w_mean=last, absolute_change=last-first,
                      percentage_change=100*(last-first)/first if first else None,
                      comparison_start=str(block.week_start.iloc[0].date()),
                      comparison_end=str((block.week_start.iloc[-1]+pd.Timedelta(days=6)).date()))
    return result


def run():
    state, inventory = partition_tickets()
    raw_cameras, pin = read_eric_registry()
    cameras = attach_identity(raw_cameras)
    n = len(cameras)
    width = int((CUTOFF - BASE_WEEK).days//7) + 1
    counts = np.zeros((n, width), dtype=np.int64)
    all_totals = np.zeros(n, dtype=np.int64)
    before_totals = np.zeros(n, dtype=np.int64)
    first_days = np.full(n, 1000000, dtype=np.int64)
    last_days = np.full(n, -1000000, dtype=np.int64)
    all_last_days = np.full(n, -1000000, dtype=np.int64)
    counties, directions = defaultdict(set), defaultdict(set)
    audit = Counter(state["counts"])
    unmatched = Counter()
    lookup = cameras[["search_query", "camera_type", "camera_index", "activation"]]
    for bucket in range(PARTITIONS):
        frame = pd.read_parquet(WORK/f"raw_{bucket:02d}.parquet")
        selected, duplicates = deduplicate(frame)
        audit["duplicate_rows_removed"] += len(frame)-len(selected)
        audit["unique_identifiers"] += len(selected)
        if len(duplicates):
            duplicates.to_parquet(OUT/f"duplicate_audit_{bucket:02d}.parquet", index=False)
            audit["conflicting_duplicate_identifiers"] += duplicates.loc[duplicates.conflicting_values, "summons_number"].nunique()
        selected = selected.copy()
        selected["date"] = pd.to_datetime(selected.issue_date, errors="coerce", format="ISO8601").dt.normalize()
        invalid = selected.date.isna()
        audit["invalid_or_missing_date_records"] += int(invalid.sum())
        if invalid.any():
            selected[invalid].to_parquet(OUT/f"invalid_dates_{bucket:02d}.parquet", index=False)
        frame = selected[~invalid & selected.date.ge(START) & selected.date.lt(END)].copy()
        audit["out_of_range_records"] += len(selected)-int(invalid.sum())-len(frame)
        frame["camera_type"] = pd.to_numeric(frame.violation_code, errors="coerce").map({7:"Red Light", 36:"Speed"})
        has_text = frame.street_name.notna() & frame.intersecting_street.notna()
        frame["search_query"] = eric_address(frame.street_name, frame.intersecting_street).where(has_text)
        frame = frame.merge(lookup, on=["search_query", "camera_type"], how="left", validate="many_to_one")
        bad = frame.camera_index.isna()
        audit["unmatched_records"] += int(bad.sum())
        for (query, kind), amount in frame.loc[bad].fillna({"search_query":"[missing address]", "camera_type":"Unknown"}).groupby(["search_query","camera_type"]).size().items():
            unmatched[(query,kind)] += int(amount)
        frame = frame[~bad].copy()
        ix = frame.camera_index.astype(int).to_numpy()
        np.add.at(all_totals, ix, 1)
        np.maximum.at(all_last_days, ix, (frame.date-pd.Timestamp("1970-01-01")).dt.days.to_numpy())
        audit["matched_records_all_dates"] += len(frame)
        for i, county in frame[["camera_index","violation_county"]].dropna().drop_duplicates().itertuples(index=False,name=None):
            counties[int(i)].add(county)
        address_variants = frame[["camera_index","street_name","intersecting_street"]].drop_duplicates()
        text = (address_variants.street_name.fillna("")+" "+address_variants.intersecting_street.fillna("")).str.upper()
        import re
        for i, address in zip(address_variants.camera_index, text):
            directions[int(i)].update(d.replace("/","") for d in re.findall(r"\b(NB|SB|EB|WB|N/B|S/B|E/B|W/B)\b",address))
        before = frame.date.lt(frame.activation)
        np.add.at(before_totals, frame.loc[before,"camera_index"].astype(int).to_numpy(), 1)
        frame = frame[~before]
        independent = frame.groupby("camera_index").agg(ticket_count=("summons_number","size"),
            first_actual_date=("date","min"),last_actual_date=("date","max")).reset_index()
        independent["camera_id"] = independent.camera_index.astype(int).map(cameras.set_index("camera_index").camera_id)
        independent.to_parquet(OUT/f"partition_group_checks_{bucket:02d}.parquet",index=False)
        ix = frame.camera_index.astype(int).to_numpy()
        weeks = ((frame.date - BASE_WEEK).dt.days//7).to_numpy()
        days = (frame.date-pd.Timestamp("1970-01-01")).dt.days.to_numpy()
        np.add.at(counts, (ix, weeks), 1)
        np.minimum.at(first_days, ix, days)
        np.maximum.at(last_days, ix, days)
        print(f"Processed identifier partition {bucket+1}/{PARTITIONS}", flush=True)
    complete = (len(inventory) == len(SOURCES) and all(m.get("complete") and not m.get("source_changed_during_pull") for m in inventory))
    # FY2017 is the earliest requested file. Earlier backdated records are not
    # evidence of complete January-June 2016 coverage without FY2016.
    coverage_weeks = pd.date_range(BASE_WEEK, periods=width, freq="W-MON")
    coverage = np.full(width, complete) & (coverage_weeks >= pd.Timestamp("2016-07-01"))
    frames, summaries = [], []
    for i, row in cameras.iterrows():
        last = pd.Timestamp("1970-01-01")+pd.Timedelta(days=int(last_days[i])) if last_days[i]>-1000000 else pd.NaT
        last_all = pd.Timestamp("1970-01-01")+pd.Timedelta(days=int(all_last_days[i])) if all_last_days[i]>-1000000 else pd.NaT
        weekly = make_weekly(row.camera_id, row.activation, last, counts[i], coverage)
        if not weekly.empty:
            frames.append(weekly)
        warnings = []
        if not (40.45 <= row.Latitude <= 40.95 and -74.30 <= row.Longitude <= -73.65):
            warnings.append("Coordinates outside NYC bounding box")
        if len(counties[i])>1:
            warnings.append("Multiple county labels")
        if len(directions[i])>1:
            warnings.append("Multiple recorded directions merged by Eric")
        group_covered = bool(complete and (weekly.empty or weekly.source_complete.all()))
        status = ("No matched tickets" if not all_totals[i] else "No tickets after activation" if weekly.empty
                  else "Matched" if group_covered else "Incomplete historical coverage")
        record = row.drop(labels=["camera_index","activation"]).to_dict()
        record.update(data_status=status, quality_flags="; ".join(warnings),
                      new_total_all_dates=int(all_totals[i]), tickets_before_activation=int(before_totals[i]),
                      analysis_ticket_count=int(counts[i].sum()), source_complete=group_covered,
                      unavailable_weeks=int((~weekly.source_complete).sum()) if not weekly.empty else 0,
                      first_actual_ticket_date=(str((pd.Timestamp("1970-01-01")+pd.Timedelta(days=int(first_days[i]))).date()) if first_days[i]<1000000 else None),
                      last_actual_ticket_date=str(last.date()) if pd.notna(last) else None,
                      new_last_ticket_date_all_dates=str(last_all.date()) if pd.notna(last_all) else None,
                      observed_weeks=len(weekly), observation_days=(last-row.activation).days+1 if pd.notna(last) else 0,
                      zero_ticket_weeks=int(weekly.ticket_count.eq(0).sum()) if not weekly.empty else 0,
                      count_difference_from_eric=int(all_totals[i])-int(row.total_tickets_issued),
                      last_week_changed=bool(pd.notna(last_all) and last_all.to_period("W-SUN").start_time != pd.Timestamp(row.last_ticket_date)),
                      county_labels="; ".join(sorted(counties[i])), recorded_directions="; ".join(sorted(directions[i])),
                      **comparison_windows(weekly))
        summaries.append(record)
    weekly = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    weekly.to_parquet(OUT/"camera_weekly_trends.parquet", index=False)
    weekly.to_csv(OUT/"camera_weekly_trends.csv.gz", index=False, compression="gzip")
    summary.to_csv(OUT/"camera_summary.csv", index=False)
    pd.DataFrame([{"search_query":k[0],"camera_type":k[1],"ticket_count":v} for k,v in unmatched.items()]).to_csv(OUT/"unmatched_addresses.csv", index=False)
    audit.update(analysis_ticket_count=int(counts.sum()), tickets_before_activation=int(before_totals.sum()))
    save_json(OUT/"reconciliation.json", {**dict(audit), "all_sources_complete":complete,
                                         "camera_groups":n, "weekly_rows":len(weekly),
                                         "source_counts":inventory, "registry_pin":pin,
                                         "full_calendar_window_covered":False,
                                         "coverage_note":"All requested FY2017-FY2027 files are checked separately. Weeks before July 4, 2016 are unavailable for complete-week totals because FY2016 is outside the requested source set. Observed early tickets remain in observed_ticket_count and analysis totals."})
    save_json(OUT/"registry_pin.json", pin)
    print(json.dumps({**dict(audit), "camera_groups":n, "all_sources_complete":complete}, indent=2), flush=True)
    return summary, weekly


if __name__ == "__main__":
    run()
