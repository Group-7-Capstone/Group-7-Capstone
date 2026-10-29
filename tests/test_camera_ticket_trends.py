"""Synthetic edge cases for the approved observation-window semantics."""
import numpy as np
import pandas as pd
import pytest

from preprocessing.camera_ticket_trends import (
    BASE_WEEK, attach_identity, comparison_windows, deduplicate, eric_address, make_weekly,
)


def test_exact_eric_normalization_preserves_known_split_word_limitation():
    street = pd.Series(["WB W 230TH ST @ TIBB", "EASTERN PKWY (W/B) @", "NB A ST @"])
    cross = pd.Series(["ET AVE", "BEDFORD AVE", "B ST"])
    assert eric_address(street, cross).tolist() == [
        "W 230TH ST AND TIBB ET AVE, NEW YORK, NY",
        "EASTERN PKWY AND BEDFORD AVE, NEW YORK, NY", "A ST AND B ST, NEW YORK, NY"]


def test_direction_variants_merge_but_types_have_distinct_ids():
    names = eric_address(pd.Series(["NB A @", "SB A @"]), pd.Series(["B", "B"]))
    assert names.nunique() == 1
    registry = pd.DataFrame({"search_query":names, "camera_type":["Speed","Red Light"],
                             "inferred_activation_date":["2021-01-04"]*2})
    assert attach_identity(registry).camera_id.nunique() == 2


def test_cross_year_duplicate_precedence_and_conflict_audit():
    base = dict(issue_date="2020-01-06",violation_code="36",street_name="A",intersecting_street="B",
                violation_county="K",issuing_agency="V")
    records = [dict(base,summons_number="1",source_fiscal_year=2020,source_row_id="a"),
               dict(base,summons_number="1",source_fiscal_year=2021,source_row_id="b",street_name="C"),
               dict(base,summons_number="2",source_fiscal_year=2020,source_row_id="d"),
               dict(base,summons_number="2",source_fiscal_year=2020,source_row_id="e")]
    selected, audit = deduplicate(pd.DataFrame(records))
    assert selected.source_row_id.tolist() == ["b", "e"]
    assert audit.loc[audit.summons_number.eq("1"), "conflicting_values"].all()
    assert not audit.loc[audit.summons_number.eq("2"), "conflicting_values"].any()
    assert audit.selected_record.sum() == 2


def weekly(start="2021-01-04", days=181, counts=None, coverage=None):
    return make_weekly("cam_test", pd.Timestamp(start), pd.Timestamp(start)+pd.Timedelta(days=days),
                       np.ones(600, dtype=int) if counts is None else counts, coverage)


def test_cross_year_week_and_partial_final_week():
    result = weekly("2020-12-28", 9)
    assert result.week_start.dt.strftime("%Y-%m-%d").tolist() == ["2020-12-28","2021-01-04"]
    assert result.is_complete_week.tolist() == [True,False]
    assert result.weeks_since_activation.tolist() == [0,1]


def test_zero_is_a_count_but_missing_coverage_breaks_line_and_average():
    values, coverage = np.zeros(600, dtype=int), np.ones(600, dtype=bool)
    ix = int((pd.Timestamp("2021-01-04")-BASE_WEEK).days//7)
    values[ix:ix+9] = [10,0,10,0,10,10,10,10,10]
    coverage[ix+4] = False
    result = weekly(days=62, counts=values, coverage=coverage)
    assert result.ticket_count.iloc[1] == 0
    assert result.moving_average_4w.iloc[3] == 5
    assert np.isnan(result.ticket_count.iloc[4])
    assert result.moving_average_4w.iloc[4:8].isna().all()
    assert result.moving_average_4w.iloc[8] == 10


def test_partial_week_does_not_enter_moving_average():
    result = weekly(days=23)
    assert result.is_partial_week.iloc[-1]
    assert result.moving_average_4w.isna().all()


def test_comparison_requires_26_full_weeks_and_zero_baseline_is_not_infinite():
    assert comparison_windows(weekly(days=180))["percentage_change"] is None
    assert comparison_windows(weekly(days=181))["percentage_change"] == 0
    result = comparison_windows(weekly(days=181, counts=np.zeros(600,dtype=int)))
    assert result["first_13w_mean"] == 0
    assert result["percentage_change"] is None


def test_no_post_activation_records_produces_empty_curve():
    assert make_weekly("empty",pd.Timestamp("2021-01-04"),pd.NaT,np.zeros(600)).empty
    assert make_weekly("empty",pd.Timestamp("2021-01-04"),pd.Timestamp("2020-12-31"),np.zeros(600)).empty


def test_identity_collision_or_duplicate_source_key_is_rejected():
    source = pd.DataFrame({"search_query":["A","A"],"camera_type":["Speed","Speed"],
                           "inferred_activation_date":["2020-01-06"]*2})
    with pytest.raises(AssertionError):
        attach_identity(source)


def test_pipeline_reconciles_duplicates_unmatched_invalid_and_empty_groups(tmp_path, monkeypatch):
    import preprocessing.camera_ticket_trends as module
    from data.camera_ticket_snapshot import SOURCES
    work, out = tmp_path/"work", tmp_path/"out"
    work.mkdir(); out.mkdir()
    common=dict(street_name="NB A @",intersecting_street="B",violation_code="36",violation_county="K",issuing_agency="V",source_fiscal_year=2021)
    specs=[("1","2021-01-04","36"),("1","2021-01-04","36"),("2","2021-01-18","36"),
           ("3","2020-12-30","36"),("4","2021-01-04","36"),("5","invalid","36"),("6","2020-12-30","7")]
    raw=pd.DataFrame([dict(common,summons_number=id_,issue_date=d,violation_code=code,source_row_id=str(i)) for i,(id_,d,code) in enumerate(specs)])
    raw.loc[0,"source_fiscal_year"]=2020
    raw.loc[2,"street_name"]="SB A @"
    raw.loc[4,"street_name"]="UNKNOWN @"
    raw.to_parquet(work/"raw_00.parquet",index=False)
    original=pd.DataFrame([dict(index=i,search_query="A AND B, NEW YORK, NY",camera_type=kind,violation_code=code,
        Latitude=40.75,Longitude=-73.95,inferred_activation_date="2021-01-04",first_ticket_date="2020-12-28",
        last_ticket_date="2021-01-18",total_tickets_issued=3,detection_confidence="low_fallback",activation_year=2021)
        for i,(kind,code) in enumerate([("Speed",36),("Red Light",7)])])
    state={"counts":{"source_records":8,"missing_identifier_records":1}}
    inventory=[{"complete":True,"source_changed_during_pull":False} for _ in SOURCES]
    monkeypatch.setattr(module,"WORK",work); monkeypatch.setattr(module,"OUT",out); monkeypatch.setattr(module,"PARTITIONS",1)
    monkeypatch.setattr(module,"partition_tickets",lambda *args, **kwargs:(state,inventory))
    monkeypatch.setattr(module,"read_camera_data",lambda name:original)
    monkeypatch.setattr(module,"camera_data_receipt",lambda name:{"sha256":"test","lf_sha256":"test"})
    summary, weeks=module.run()
    speed=summary[summary.camera_type.eq("Speed")].iloc[0]
    red=summary[summary.camera_type.eq("Red Light")].iloc[0]
    assert len(summary)==2 and not weeks.duplicated(["camera_id","week_start"]).any()
    assert weeks.observed_ticket_count.tolist()==[1,0,1]
    assert speed.analysis_ticket_count==2 and speed.new_total_all_dates==3
    assert speed.last_actual_ticket_date=="2021-01-18"
    assert red.data_status=="No tickets after activation" and red.analysis_ticket_count==0
    assert "Multiple recorded directions" in speed.quality_flags
    audit=module.json.loads((out/"reconciliation.json").read_text())
    assert audit["source_records"]==audit["unique_identifiers"]+audit["duplicate_rows_removed"]+audit["missing_identifier_records"]
    assert audit["unique_identifiers"]==audit["analysis_ticket_count"]+audit["tickets_before_activation"]+audit["unmatched_records"]+audit["invalid_or_missing_date_records"]
