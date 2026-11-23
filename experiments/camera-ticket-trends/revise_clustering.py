"""Reproduce #58 locally from the rebuilt ticket snapshot; never downloads or publishes."""
import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

from data.reader import LION_PATH, NYCWALKS_PATH, camera_data_receipt, read_camera_data
from preprocessing.camera_corrected_trends import (
    correct_calendar_weeks,
    first_year_outcomes,
)
from preprocessing.camera_shared_trends import DEFAULTS, load_inputs
from preprocessing.camera_trend_sites import load_ticket_site_features
from preprocessing.config import TICKET_REVISION_CONFIG
from validation.camera_corrected_trends import (
    camera_halves,
    cluster_feature_profiles,
    compare_feature_groups,
    discover_clusters,
    feature_rules,
)


def checksum(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def run(input_dir, output_dir, reuse_features=False, established_weeks=52):
    config = {**DEFAULTS, **TICKET_REVISION_CONFIG, "established_weeks": established_weeks}
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, weekly, audit = load_inputs(input_dir)
    current = camera_data_receipt("camera_activation_dates")
    if audit["registry_pin"]["sha256"] not in (current["sha256"], current["lf_sha256"]):
        raise ValueError("Rebuild weekly tickets against the current #23 registry first")
    if not audit["all_sources_complete"]:
        raise ValueError("The source snapshot is incomplete")
    speed_ids = summary.loc[summary.camera_type.eq("Speed"), "camera_id"]
    split = camera_halves(speed_ids, config["seed"])
    corrected = correct_calendar_weeks(summary, weekly, read_camera_data("speed_camera_schedule"), split.index[split.eq("A")], config)
    outcomes, panel = first_year_outcomes(summary, corrected, config)
    clusters, diagnostics, selection = discover_clusters(outcomes, panel, split, config)
    feature_path = output_dir / "site_features.csv"
    if reuse_features:
        features = pd.read_csv(feature_path)
        if set(features.camera_id) != set(speed_ids) or not features.camera_id.is_unique:
            raise ValueError("Cached site features do not match this camera registry")
    else:
        features = load_ticket_site_features(summary)
        features.to_csv(feature_path, index=False)
    full = outcomes.join(split).join(clusters).join(features.set_index("camera_id"), validate="one_to_one")
    selected = full.loc[full.outcome_status.eq("eligible")].copy()
    if selection["mode"] == "clusters":
        selected = selected.loc[selected.cluster.notna()]
    rules = feature_rules(selected.loc[selected.split.eq("A")], config)
    comparisons, strata = compare_feature_groups(selected, rules, config)
    profiles = cluster_feature_profiles(selected, rules)
    cluster_tests = []
    if selection["mode"] == "clusters":
        for cluster in sorted(selected.cluster.dropna().unique()):
            # For two clusters the second contrast is just the sign reversal.
            if selection["selected_k"] == 2 and cluster != 1:
                continue
            test_frame = selected.assign(cluster_share_pct=selected.cluster.eq(cluster).astype(float) * 100)
            tests, _ = compare_feature_groups(test_frame, rules, config, response="cluster_share_pct")
            cluster_tests.append(tests.assign(cluster=int(cluster)))
    if cluster_tests:
        cluster_tests = pd.concat(cluster_tests, ignore_index=True)
        # Adjust across clusters too if A selects more than two clusters.
        cluster_tests["B_p_holm"] = (cluster_tests.B_p_holm * len(cluster_tests.cluster.unique())).clip(upper=1)
        cluster_tests["replicated"] = cluster_tests.same_direction & cluster_tests.B_p_holm.lt(config["alpha"])
        cluster_tests.to_csv(output_dir / "cluster_feature_validation.csv", index=False)
    full.to_csv(output_dir / "camera_outcomes.csv")
    panel.to_parquet(output_dir / "corrected_first_year.parquet")
    corrected.to_parquet(output_dir / "corrected_calendar_weeks.parquet", index=False)
    diagnostics.to_csv(output_dir / "cluster_diagnostics_A.csv", index=False)
    comparisons.to_csv(output_dir / "feature_validation.csv", index=False)
    strata.to_csv(output_dir / "feature_validation_strata.csv", index=False)
    profiles.to_csv(output_dir / "cluster_feature_profiles.csv", index=False)
    selected.groupby(["split", "cluster"]).decline_52w_pct.agg(["count", "median", "mean"]).to_csv(output_dir / "cluster_outcomes.csv")
    full.groupby(["split", "activation_year", "outcome_status"]).size().rename("cameras").to_csv(output_dir / "eligibility_by_year.csv")
    corrected.groupby("week_start").agg(usable_camera_weeks=("corrected_rate", "count"),
        minimum_reference_n=("reference_count", "min"), maximum_reference_n=("reference_count", "max")).to_csv(output_dir / "calendar_reference_coverage.csv")
    checks = {
        "all_source_records_accounted": audit["source_records"] == audit["unique_identifiers"] + audit["duplicate_rows_removed"] + audit["missing_identifier_records"],
        "all_unique_tickets_accounted": audit["unique_identifiers"] == sum(audit[k] for k in ["analysis_ticket_count", "tickets_before_activation", "unmatched_records", "invalid_or_missing_date_records", "out_of_range_records"]),
        "all_speed_cameras_retained": set(full.index) == set(speed_ids),
        "split_is_disjoint": set(split.index[split.eq("A")]).isdisjoint(split.index[split.eq("B")]),
        "no_censored_outcomes": full.loc[full.left_censored, "decline_52w_pct"].isna().all(),
        "no_zero_hours_used": corrected.loc[corrected.permitted_hours.eq(0), "corrected_rate"].isna().all(),
        "minimum_reference_size": corrected.loc[corrected.corrected_rate.notna(), "reference_count"].ge(config["minimum_references"]).all(),
        "feature_keys_unique": features.camera_id.is_unique,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    if not all(checks.values()):
        raise AssertionError(checks)
    inputs = {name: checksum(input_dir / name) for name in ["camera_summary.csv", "camera_weekly_trends.parquet", "reconciliation.json"]}
    spatial = {"lion": {p.name: checksum(p) for p in sorted(LION_PATH.iterdir()) if p.is_file() and not p.name.endswith(".lock")},
               "nycwalks": {p.name: checksum(p) for p in sorted(NYCWALKS_PATH.parent.glob(NYCWALKS_PATH.stem + ".*")) if p.is_file()}}
    code_paths = ["preprocessing/config.py", "preprocessing/camera_corrected_trends.py", "preprocessing/camera_trend_sites.py",
                  "preprocessing/cameras.py", "preprocessing/camera_locations.py", "preprocessing/roadway.py", "preprocessing/pedestrians.py",
                  "preprocessing/camera_shared_trends.py", "model/camera_shape.py", "validation/camera_shared_trends.py",
                  "validation/camera_corrected_trends.py", "experiments/camera-ticket-trends/revise_clustering.py"]
    result = {"issue": 58, "config": config, "selection": selection, "feature_rules_from_A": rules,
              "speed_cameras": len(speed_ids), "split_counts": split.value_counts().to_dict(),
              "outcome_status": full.outcome_status.value_counts().to_dict(),
              "outcomes_by_half": full.loc[full.outcome_status.eq("eligible")].groupby("split").decline_52w_pct.agg(["count", "median", "mean"]).to_dict("index"),
              "feature_coverage": features[["direction_pair", "lanes", "width_ft", "ped_volume", "borough"]].notna().sum().to_dict(),
              "checks": checks, "input_sha256": inputs, "registry": current,
              "schedule": camera_data_receipt("speed_camera_schedule"), "spatial_sha256": spatial,
              "code_sha256": {name: checksum(ROOT / name) for name in code_paths},
              "interpretation": "Descriptive ticket trends, not causal crash effects. Positive decline means fewer tickets relative to established A cameras. Fixed road/pedestrian snapshots are not historical exposure."}
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["selection", "outcome_status", "outcomes_by_half", "feature_coverage", "checks"]}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "experiments/camera-ticket-trends/outputs/revised_inputs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/camera-ticket-trends/outputs/issue58")
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--established-weeks", type=int, default=TICKET_REVISION_CONFIG["established_weeks"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(args.input_dir, args.output_dir, args.reuse_features, args.established_weeks)
