"""Build first-year ticket trends by unordered pairs of intersecting streets."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from data.reader import read_lion_road_attributes
from preprocessing.camera_intersection_pairs import PAIR_CONFIG, PAIR_DIMENSIONS, classify_pairs
from preprocessing.camera_road_features import UNKNOWN
from preprocessing.camera_shared_trends import DEFAULTS, aligned_panels, load_inputs, normalize_baseline
from validation.camera_shared_trends import describe_matrix


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run(gdb, inputs, out):
    out.mkdir(parents=True, exist_ok=True)
    frozen = inputs / "road_features"
    prior = json.loads((frozen / "lion_receipt.json").read_text(encoding="utf-8"))
    roads, aliases, receipt = read_lion_road_attributes(gdb)
    if prior["files"] != receipt["files"]:
        raise ValueError("The supplied LION snapshot differs from the first-street reference")
    summary, weekly, audit = load_inputs(inputs)
    first = pd.read_csv(frozen / "camera_road_features.csv")
    segments = pd.read_csv(frozen / "matched_segments.csv", dtype={"segment_id":str,"node_from":str,"node_to":str})
    if set(summary.camera_id) != set(first.camera_id) or not summary.camera_id.is_unique:
        raise ValueError("Camera registries differ")
    print(f"Classifying road pairs for {first.strict_match.sum():,} original intersection matches.", flush=True)
    directory = classify_pairs(first, segments, roads, aliases).set_index("camera_id")
    del roads, aliases
    paired = directory.loc[directory.pair_status.eq("Two streets matched")]
    pair_cols = [d+"_pair" for d in PAIR_DIMENSIONS]
    if paired.groupby("intersection_pair_id")[pair_cols].nunique().gt(1).any().any():
        raise ValueError("Inconsistent classification of the same two roads at a node")
    print("Aligning complete ticket histories.", flush=True)
    _, primary, _, _ = aligned_panels(summary, weekly, audit)
    n_weeks, baseline = DEFAULTS["primary_horizon"], DEFAULTS["baseline_weeks"]
    full = primary.loc[:, 1:n_weeks].dropna()
    directory["eligible_52w"] = directory.index.isin(full.index)
    reference = pd.read_csv(inputs / "shared_trends" / "camera_eligibility.csv").set_index("camera_id")
    if not directory.eligible_52w.equals(reference.eligible_52w.reindex(directory.index)):
        raise ValueError("First-year membership differs from the frozen analysis")
    eligible = directory.loc[full.index]
    traces, results = [], []
    for kind, kind_sites in eligible.groupby("camera_type"):
        for dim in PAIR_DIMENSIONS:
            for category, sites in kind_sites.groupby(dim+"_pair", sort=True):
                raw = full.loc[sites.index]
                idx = normalize_baseline(raw, list(range(1, baseline+1)))
                ids = sites.intersection_pair_id.replace("", np.nan)
                keys = dict(camera_type=kind, dimension=dim, road_pair=category)
                curves = describe_matrix(raw, idx, kind, "intersection_pairs", DEFAULTS)
                traces.append(curves.rename(columns={"time":"week"}).drop(columns=["scenario","scenario_label"]).assign(**keys))
                late = idx.iloc[:, -baseline:].mean(axis=1)
                raw_sum = int(raw.sum().sum())
                results.append({**keys, "n":len(sites), "index_n":int(late.notna().sum()),
                    "known_named_pairs":ids.nunique(), "known_nodes":sites.node_id.replace("",np.nan).nunique(),
                    "small_group":len(sites)<PAIR_CONFIG["small_group_n"], "tickets_52w":raw_sum,
                    "first_13w_raw_mean":raw.iloc[:, :baseline].mean().mean(),
                    "last_13w_raw_mean":raw.iloc[:, -baseline:].mean().mean(),
                    "last_13w_index_mean":late.mean(), "mean_change_pct":late.mean()-100,
                    "median_location_change_pct":late.median()-100,
                    "equal_node_change_pct":late.groupby(sites.node_id.replace("",np.nan)).mean().mean()-100
                        if category != UNKNOWN else np.nan})
    coverage, trends = pd.DataFrame(results), pd.concat(traces, ignore_index=True)
    if trends.duplicated(["camera_type","dimension","road_pair","week"]).any():
        raise ValueError("Duplicate weekly key")
    for (kind, dim), groups in coverage.groupby(["camera_type","dimension"]):
        raw = full.loc[eligible.index[eligible.camera_type.eq(kind)]]
        if len(raw) != groups.n.sum() or int(raw.sum().sum()) != groups.tickets_52w.sum():
            raise ValueError(f"Coverage reconciliation failed: {kind}/{dim}")
        rows = trends.loc[trends.camera_type.eq(kind) & trends.dimension.eq(dim)]
        weekly_total = (rows.raw_mean*rows.n).groupby(rows.week).sum()
        np.testing.assert_allclose(weekly_total,raw.sum().to_numpy(),atol=1e-7,rtol=1e-12)
    np.testing.assert_allclose(trends.loc[trends.week.le(baseline)].groupby(
        ["camera_type","dimension","road_pair"]).index_mean.mean().dropna(),100,atol=1e-8)
    directory.to_csv(out/"camera_pair_directory.csv",index_label="camera_id")
    physical = paired.groupby("intersection_pair_id").agg(node_id=("node_id","first"),
        node_level=("node_level","first"),address_example=("search_query","first"),
        all_camera_groups=("search_query","size"),**{c:(c,"first") for c in pair_cols})
    physical.to_csv(out/"named_intersections.csv")
    coverage.to_csv(out/"pair_summary.csv",index=False)
    trends.to_csv(out/"pair_weekly_trends.csv",index=False)
    directory.groupby(["camera_type","eligible_52w","pair_status"]).size().rename("n").to_csv(out/"match_coverage.csv")
    sources = [inputs / "camera_summary.csv", inputs / "camera_weekly_trends.parquet",
               inputs / "reconciliation.json", inputs / "shared_trends" / "camera_eligibility.csv", frozen / "camera_road_features.csv",
               frozen / "matched_segments.csv", frozen / "lion_receipt.json"]
    manifest = {"original_groups":len(directory),"cohort":eligible.camera_type.value_counts().to_dict(),
                "both_streets_matched":len(paired),"unique_named_pairs":len(physical),
                "unique_node_levels":len(paired[["node_id","node_level"]].drop_duplicates()),
                "method":"One unique common LION node and level within 60 m; distinct, ordinary undivided street segments.",
                "confidence":"Named-street and node match. Enforced approach and historical road conditions are unverified.",
                "width":"A street is narrow when all valid min/max widths are <15 m; wide when all are ≥15 m. Other cases remain unknown.",
                "lanes":"Consensus travel-lane bands: 1, 2, 3 and 4+. Two-way counts include both directions.",
                "unit":"One registry address/type group per category and relative week; order of the two streets is immaterial.",
                "period":"First 52 complete Monday–Sunday weeks after inferred activation, through last ticket; issue-date cutoff 2026-08-30.",
                "change":"Equal-location mean of weeks 40–52 relative to own weeks 1–13. Zero baselines excluded from indices.",
                "equal_node_change_pct":"Sensitivity only: average location changes within each matched node, then average nodes equally.",
                "uncertainty":"Pointwise 95% bootstrap of 500 whole location histories; shared-node correlations, matching and exposure are unmodeled.",
                "unresolved":"All original groups retained; attribute ambiguity on an otherwise matched pair is dimension-specific.",
                "input_sha256":{str(path.relative_to(inputs)):file_hash(path) for path in sources},
                "validation":"Stable frozen cohort, unique weekly keys, constant intersection class, baseline 100, and per-type weekly counts reconciled in every dimension."}
    (out/"methods.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(coverage[["camera_type","dimension","road_pair","n","known_nodes","mean_change_pct"]].to_string(index=False),flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lion-gdb",type=Path,required=True)
    p.add_argument("--inputs",type=Path,default=HERE/"outputs")
    p.add_argument("--output",type=Path,default=HERE/"outputs"/"intersection_pairs")
    args = p.parse_args()
    with threadpool_limits(limits=2):
        run(args.lion_gdb,args.inputs,args.output)
