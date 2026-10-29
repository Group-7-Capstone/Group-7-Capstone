"""Finalize local browser state after numerical and rendered acceptance checks."""
from pathlib import Path
import json
import hashlib

HERE=Path(__file__).resolve().parent
OUT=HERE/"outputs"


def main(review_path):
    validation=json.loads((OUT/"artifact_validation.json").read_text(encoding="utf-8"))
    assert validation["browser_values_match_parquet"] and validation["notebook_error_outputs"]==0
    data_path=HERE/"browser/src/data.json"
    data=json.loads(data_path.read_text(encoding="utf-8"))
    data["buildStatus"]="complete"
    data_path.write_text(json.dumps(data,ensure_ascii=False,separators=(",",":"),allow_nan=False),encoding="utf-8")
    qa = json.loads(Path(review_path).read_text(encoding="utf-8"))
    required = {"full_repository_tests_passed", "browser_checks", "rendered_review"}
    if not required.issubset(qa):
        raise ValueError("Supply an actual integration review record, not historical QA claims")
    qa.update(validation)
    qa.update(outputs_local_only=True, github_push_performed=False)
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-record", required=True, type=Path)
    args = parser.parse_args()
    main(args.review_record)
