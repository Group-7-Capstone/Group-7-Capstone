"""Reconcile the frozen registry, source receipts, and final weekly series."""
import json
import numpy as np
import pandas as pd
from data.reader import read_eric_registry
from data.camera_ticket_snapshot import save_json
from preprocessing.camera_ticket_trends import OUT, attach_identity


def validate():
    summary = pd.read_csv(OUT/"camera_summary.csv")
    weekly = pd.read_parquet(OUT/"camera_weekly_trends.parquet")
    audit = json.loads((OUT/"reconciliation.json").read_text())
    original, pin = read_eric_registry()
    original = attach_identity(original).set_index("camera_id")
    checks = []

    def check(name, ok):
        checks.append({"check":name,"passed":bool(ok)})

    check("All 4,610 source groups retained",len(summary)==len(original)==4610)
    check("Unique camera identifiers",summary.camera_id.is_unique)
    check("Unique camera-week keys",not weekly.duplicated(["camera_id","week_start"]).any())
    check("Source checksum preserved",pin==audit["registry_pin"])
    check("Raw identifier reconciliation",audit["source_records"]==audit.get("missing_identifier_records",0)+audit["unique_identifiers"]+audit["duplicate_rows_removed"])
    check("Unique ticket disposition reconciliation",audit["unique_identifiers"]==sum(audit.get(k,0) for k in ["invalid_or_missing_date_records","out_of_range_records","unmatched_records","tickets_before_activation","analysis_ticket_count"]))
    check("All source totals reconcile",audit["source_records"]==sum(sum(m["actual"].values()) for m in audit["source_counts"]))
    check("Snapshot sources complete and stable",audit["all_sources_complete"])
    check("Weekly total reconciles",int(weekly.observed_ticket_count.sum())==audit["analysis_ticket_count"])
    grouped = dict(tuple(weekly.groupby("camera_id")))
    raw_checks = pd.concat([pd.read_parquet(p) for p in sorted(OUT.glob("partition_group_checks_*.parquet"))])
    raw_checks = raw_checks.groupby("camera_id").agg(tickets=("ticket_count","sum"),last_date=("last_actual_date","max"))
    for row in summary.itertuples():
        check(f"{row.camera_id}: unchanged activation",row.inferred_activation_date==original.loc[row.camera_id,"inferred_activation_date"])
        check(f"{row.camera_id}: unchanged original attributes",all(str(getattr(row,k))==str(original.loc[row.camera_id,k]) for k in ["search_query","camera_type","detection_confidence","Latitude","Longitude"]))
        frame = grouped.get(row.camera_id)
        if frame is None:
            check(f"{row.camera_id}: empty count",row.analysis_ticket_count==0)
            continue
        check(f"{row.camera_id}: total",frame.observed_ticket_count.sum()==row.analysis_ticket_count)
        check(f"{row.camera_id}: independent raw total",row.analysis_ticket_count==raw_checks.loc[row.camera_id,"tickets"])
        check(f"{row.camera_id}: raw maximum actual date",pd.Timestamp(row.last_actual_ticket_date)==raw_checks.loc[row.camera_id,"last_date"])
        check(f"{row.camera_id}: last week",frame.week_start.max()==pd.Timestamp(row.last_actual_ticket_date).to_period("W-SUN").start_time)
        check(f"{row.camera_id}: ordered weekly sequence",frame.week_start.diff().dropna().eq(pd.Timedelta(days=7)).all())
        expected = frame.ticket_count.where(frame.is_complete_week).rolling(4,min_periods=4).mean()
        check(f"{row.camera_id}: moving average",np.allclose(frame.moving_average_4w,expected,equal_nan=True))
        check(f"{row.camera_id}: missingness",frame.loc[~frame.source_complete,"ticket_count"].isna().all())
    results = pd.DataFrame(checks)
    results.to_csv(OUT/"validation_checks.csv",index=False)
    save_json(OUT/"validation_summary.json",{"checks":len(checks),"passed":int(results.passed.sum()),
                                             "failed":results.loc[~results.passed,"check"].tolist()})
    if not results.passed.all():
        raise AssertionError(results.loc[~results.passed].to_string(index=False))
    print(f"Passed {len(checks):,} reconciliation checks.")
    return results


if __name__ == "__main__":
    validate()
