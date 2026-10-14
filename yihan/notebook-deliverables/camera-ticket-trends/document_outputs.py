"""Write field definitions, source coverage, and a compact local result ledger."""
from pathlib import Path
import json
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/"outputs"
DEFINITIONS={
    "index":"Original Eric registry row index, preserved.",
    "search_query":"Eric's unchanged normalized address string used as an exact matching key.",
    "camera_type":"Original camera type: Speed or Red Light. Part of the group key.",
    "violation_code":"Original violation code, preserved: 36 for speed, 7 for red light.",
    "Latitude":"Original geocoded latitude; preserved without verification or correction.",
    "Longitude":"Original geocoded longitude; preserved without verification or correction.",
    "inferred_activation_date":"Eric's original inferred activation date. Not a verified installation date.",
    "first_ticket_date":"Eric's original first observed week-start date, preserved.",
    "last_ticket_date":"Eric's original last observed week-start date, preserved for comparison.",
    "total_tickets_issued":"Eric's original cumulative ticket count, preserved.",
    "detection_confidence":"Eric's unchanged confidence label: high_sustained or low_fallback.",
    "activation_year":"Eric's original activation-year label, preserved.",
    "camera_id":"Stable cam_ identifier: first 16 SHA-256 hex digits of exact address, NUL separator, and camera type.",
    "data_status":"Matched, Incomplete historical coverage, No matched tickets, or No tickets after activation.",
    "quality_flags":"Automated flags for inherited grouping/coordinate limitations; no group is removed.",
    "new_total_all_dates":"Unique matching tickets in the entire issue-date extraction range, including pre-activation tickets.",
    "tickets_before_activation":"Unique matching tickets before Eric's unchanged activation date, excluded from the plotted observation window.",
    "analysis_ticket_count":"Unique matched tickets on/after activation and within the extraction range; includes observed tickets in incomplete-coverage weeks. Equals sum of observed_ticket_count.",
    "source_complete":"Weekly table: full-week source coverage available. Summary: every week of that group's observation window has source coverage.",
    "unavailable_weeks":"Number of weeks with unavailable full-week source coverage; these have null ticket_count.",
    "first_actual_ticket_date":"Earliest actual daily ticket date on/after activation in the new extraction; null for an empty history.",
    "last_actual_ticket_date":"Latest actual daily ticket date on/after activation in the new extraction; null for an empty history. Not a deactivation date.",
    "new_last_ticket_date_all_dates":"Latest matched actual daily date anywhere in the extraction range, including pre-activation observations, for comparison with Eric's old last week.",
    "observed_weeks":"Number of Monday-start weeks in the activation-to-last-ticket observation window, including partial or unavailable weeks.",
    "observation_days":"Inclusive calendar-day span from activation to last actual ticket, or zero for an empty history.",
    "zero_ticket_weeks":"Number of source-covered observation weeks with zero observed tickets; unavailable weeks do not count as zero.",
    "count_difference_from_eric":"new_total_all_dates minus original total_tickets_issued; changes may reflect fresh-source revisions.",
    "last_week_changed":"True when the Monday containing new_last_ticket_date_all_dates differs from Eric's original last-ticket week. False when there is no matched history.",
    "county_labels":"Distinct raw county labels among all matched tickets; aliases may refer to the same borough.",
    "recorded_directions":"Distinct NB/SB/EB/WB direction tokens removed by Eric's normalization and retained as an audit attribute.",
    "first_13w_mean":"Mean tickets in the initial 13 weeks of the longest run with at least 26 complete, covered weeks; null if ineligible.",
    "last_13w_mean":"Mean tickets in the final 13 weeks of that same eligible run; does not overlap the initial window.",
    "absolute_change":"last_13w_mean minus first_13w_mean, in tickets per week.",
    "percentage_change":"100 times absolute_change divided by first_13w_mean; null for zero baseline or ineligibility.",
    "comparison_start":"Monday starting the selected longest complete run, earliest run on ties.",
    "comparison_end":"Sunday ending the selected longest complete run.",
    "week_start":"Monday date for the calendar-week bin, in local issue-date calendar semantics without a timezone conversion.",
    "weeks_since_activation":"Floor of elapsed days from activation to week_start divided by seven. Can be -1 if a non-Monday activation falls within the first calendar week.",
    "observed_ticket_count":"Actual unique tickets in that week within the group's analysis window, including partial or unavailable-coverage weeks. Audit count, not an estimate of missing tickets.",
    "ticket_count":"observed_ticket_count when source coverage is complete; otherwise null. Zero is no observed ticket, not verified device inactivity.",
    "is_complete_week":"True only when all seven days fall inside the analysis interval and full source coverage is available.",
    "moving_average_4w":"Trailing mean of ticket_count across four consecutive complete weeks. Partial/unavailable weeks reset eligibility.",
    "is_partial_week":"True when the first or last observation week contains fewer than seven days inside the analysis interval, independently of source coverage.",
}

def main():
    summary=pd.read_csv(OUT/"camera_summary.csv")
    weekly=pd.read_parquet(OUT/"camera_weekly_trends.parquet")
    audit=json.loads((OUT/"reconciliation.json").read_text())
    definitions=[]
    for filename,frame in [("camera_summary.csv",summary),("camera_weekly_trends.parquet",weekly)]:
        for field in frame.columns:
            definitions.append({"file":filename,"field":field,"dtype":str(frame[field].dtype),"definition":DEFINITIONS[field]})
    pd.DataFrame(definitions).to_csv(OUT/"field_dictionary.csv",index=False)
    sources=pd.DataFrame([{"fiscal_year":m["fiscal_year"],"dataset_id":m["dataset_id"],
        "expected_records":sum(m["expected"].values()),"downloaded_records":sum(m["actual"].values()),
        "complete":m["complete"],"source_changed_during_pull":m["source_changed_during_pull"],
        "started_at":m.get("started_at"),"finished_at":m.get("finished_at")} for m in audit["source_counts"]])
    sources.to_csv(OUT/"source_coverage.csv",index=False)
    summary.groupby(["camera_type","detection_confidence","data_status"],dropna=False).agg(
        groups=("camera_id","size"),tickets=("analysis_ticket_count","sum"),
        unavailable_weeks=("unavailable_weeks","sum")).reset_index().to_csv(OUT/"group_coverage.csv",index=False)
    ledger={"groups":len(summary),"camera_types":summary.camera_type.value_counts().to_dict(),
        "status_counts":summary.data_status.value_counts().to_dict(),"raw_records":audit["source_records"],
        "unique_identifiers":audit["unique_identifiers"],"matched_tickets_after_activation":int(summary.analysis_ticket_count.sum()),
        "weekly_rows":len(weekly),"eligible_comparisons":int(summary.first_13w_mean.notna().sum()),
        "groups_with_quality_flags":int(summary.quality_flags.fillna("").ne("").sum()),
        "groups_with_changed_totals":int(summary.count_difference_from_eric.ne(0).sum()),
        "groups_with_changed_last_week":int(summary.last_week_changed.sum()),
        "unavailable_weekly_rows":int((~weekly.source_complete).sum()),
        "observed_tickets_in_unavailable_weeks":int(weekly.loc[~weekly.source_complete,"observed_ticket_count"].sum()),
        "coverage_note":audit["coverage_note"]}
    (OUT/"result_ledger.json").write_text(json.dumps(ledger,indent=2),encoding="utf-8")
    print(json.dumps(ledger,indent=2))

if __name__=="__main__":
    main()
