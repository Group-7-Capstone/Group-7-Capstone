"""Finalize local browser state after numerical and rendered acceptance checks."""
from pathlib import Path
import json
import hashlib

HERE=Path(__file__).resolve().parent
OUT=HERE/"outputs"


def main():
    validation=json.loads((OUT/"artifact_validation.json").read_text())
    assert validation["browser_values_match_parquet"] and validation["notebook_error_outputs"]==0
    data_path=HERE/"browser/src/data.json"
    data=json.loads(data_path.read_text(encoding="utf-8"))
    data["buildStatus"]="complete"
    data_path.write_text(json.dumps(data,ensure_ascii=False,separators=(",",":"),allow_nan=False),encoding="utf-8")
    qa={"assessment":"Ready for descriptive use with documented coverage and grouping caveats",
        "full_repository_tests_passed":18,"reconciliation_checks_passed":41499,
        "browser_checks":["ID and address search","Speed and Red Light selection","Original confidence filters",
            "Data status filter returns 326 incomplete-coverage groups","Numeric ticket sorting",
            "Calendar and relative-week axis switching","CSV export: 165 rows, 447561 tickets for cam_3fba8be7f3891741",
            "PNG export: 2760 x 1440 pixels, selected group identity and definitions retained"],
        "rendered_review":"Seven static figures inspected; one-point visibility and narrow browser header corrected. Both types, both confidence labels, low/median/high volumes, and incomplete early coverage checked. Browser interactions are representative spot checks, not individual UI tests for every group.",
        "presentation_limitation":"Shared browser number/date formatting follows browser locale; authored content and static notebook figure labels are English.",
        "offline_export_check":"Verified Data offline export created. Direct file:// browser test blocked by in-app URL policy; localhost browser and its CSV/PNG exports tested.",
        "outputs_local_only":True,"github_push_performed":False,**validation}
    (OUT/"acceptance_record.json").write_text(json.dumps(qa,indent=2),encoding="utf-8")
    files=[HERE/"Per_Camera_Ticket_Trends.ipynb",HERE/"Per_Camera_Ticket_Trends.html",
        OUT/"camera_summary.csv",OUT/"camera_weekly_trends.parquet",OUT/"camera_weekly_trends.csv.gz",
        OUT/"field_dictionary.csv",OUT/"reconciliation.json",OUT/"validation_checks.csv",OUT/"registry_pin.json"]
    if (HERE/"Camera_Ticket_Trends.html").exists():
        files.append(HERE/"Camera_Ticket_Trends.html")
    hashes={str(p.relative_to(HERE)).replace("\\","/"):{"bytes":p.stat().st_size,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in files}
    (OUT/"deliverable_manifest.json").write_text(json.dumps(hashes,indent=2),encoding="utf-8")
    print("Saved final local acceptance record and delivery checksums.")


if __name__=="__main__":
    main()
