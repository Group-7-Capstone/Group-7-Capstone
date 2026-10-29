"""Reproduce shared trends, shape clusters, and the companion notebook locally."""
from pathlib import Path
import argparse
from datetime import datetime, timezone
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from preprocessing.camera_shared_trends import (
    DEFAULTS, load_inputs, aligned_panels, eligibility_table, normalize_baseline,
    descriptive_rows, shape_features, profile_cluster,
)
from validation.camera_shared_trends import (
    select_shape_model, subset_stability, validate_panels, describe_matrix,
)

OUT = HERE / "outputs/shared_trends"


def rows_json(frame):
    return json.loads(frame.to_json(orient="records", date_format="iso", double_precision=10))


def run_analysis(config=None):
    config = dict(DEFAULTS if config is None else config)
    OUT.mkdir(parents=True, exist_ok=True)
    summary, weekly, audit = load_inputs(HERE / "outputs")
    meta, primary, administrative, calendar = aligned_panels(summary, weekly, audit, config)
    eligibility = eligibility_table(meta, primary, administrative, config)
    h, baseline = config["primary_horizon"], list(range(1, config["baseline_weeks"] + 1))
    eligible = primary.loc[:, 1:h].notna().all(axis=1)
    eligibility["exclusion_reason"] = np.select(
        [eligible, meta.anchor.lt(pd.Timestamp(config["coverage_start"])),
         meta.anchor.add(pd.Timedelta(weeks=h) - pd.Timedelta(days=1)).gt(pd.Timestamp(config["cutoff"]))],
        ["Eligible", "Incomplete coverage at inferred activation", "Administrative cutoff before 52 complete weeks"],
        default="Last ticket before end of 52-week window or incomplete coverage")
    results, calendars, memberships, profiles, cluster_curves, diagnostics, sensitivities = [], [], [], [], [], [], []
    model_summary = []
    with threadpool_limits(limits=2):
        for kind in sorted(meta.camera_type.unique()):
            ids = meta.index[meta.camera_type.eq(kind)]
            first52 = primary.loc[ids, 1:h].dropna()
            scenarios = {f"balanced_{n}": primary.loc[ids, 1:n].dropna() for n in config["horizons"]}
            scenarios.update({"available": primary.loc[ids],
                              "high_confidence": first52.loc[meta.loc[first52.index].detection_confidence.eq("high_sustained")],
                              "skip_first_four": first52.loc[:, config["skip_weeks"]+1:h],
                              "administrative": administrative.loc[ids, 1:h].dropna(),
                              "baseline_at_least_five": first52.loc[first52.loc[:, baseline].mean(axis=1).ge(5)]})
            for year in sorted(meta.loc[first52.index].activation_year.unique()):
                scenarios[f"cohort_{year}"] = first52.loc[meta.loc[first52.index].activation_year.eq(year)]
            for scenario, matrix in scenarios.items():
                if matrix.empty:
                    continue
                if scenario == "available":
                    matrix = matrix.loc[:, :int(matrix.notna().any().to_numpy().nonzero()[0].max())+1]
                base = list(range(config["skip_weeks"]+1, config["skip_weeks"]+config["baseline_weeks"]+1)) if scenario == "skip_first_four" else baseline
                results.append(describe_matrix(matrix, normalize_baseline(matrix, base), kind, scenario, config))
            cal = calendar.loc[ids]
            base_mean = eligibility.loc[ids, "baseline_mean"].where(eligibility.loc[ids, "baseline_mean"].gt(0))
            calendars.append(describe_matrix(cal, cal.div(base_mean, axis=0)*100, kind, "calendar", config))
            features, constant_ids = shape_features(first52, config)
            eligibility.loc[constant_ids, "cluster_status"] = "Constant four-week block profile"
            model, diag = select_shape_model(features, config)
            diag["camera_type"] = kind
            diagnostics.append(diag)
            feature_export = features.copy()
            feature_export.columns = [f"block_{i}_z" for i in feature_export.columns]
            feature_export.to_csv(OUT / f"shape_features_{kind.lower().replace(' ', '_')}.csv", index_label="camera_id")
            if model is None:
                eligibility.loc[features.index, "cluster_status"] = "No solution passed selection thresholds"
                best = diag.sort_values("silhouette", ascending=False).iloc[0] if len(diag) else None
                model_summary.append({"camera_type": kind, "eligible_groups": len(first52), "modeled_groups": len(features), "selected_k": None,
                                      "best_candidate_k": int(best.k) if best is not None else None,
                                      "best_candidate_silhouette": float(best.silhouette) if best is not None else None,
                                      "best_candidate_stability": float(best.stability_ari) if best is not None else None,
                                      "status": "No candidate passed all selection thresholds"})
                continue
            member, profile, curve = profile_cluster(meta, first52, features, model.labels_)
            memberships.append(member)
            profiles.append(profile)
            cluster_curves.append(curve)
            eligibility.loc[features.index, "cluster_status"] = "Assigned"
            chosen = diag[diag.selected].iloc[0]
            model_summary.append({"camera_type": kind, "eligible_groups": len(first52), "modeled_groups": len(features), "selected_k": int(chosen.k), "silhouette": float(chosen.silhouette), "stability_ari": float(chosen.stability_ari), "status": "Exploratory shape partition"})
            masks = {"Exclude sparse histories": first52.index[first52.gt(0).sum(axis=1).gt(config["sparse_max_nonzero_weeks"])],
                     "High sustained confidence only": meta.index[meta.detection_confidence.eq("high_sustained")]}
            for label, subset in masks.items():
                sensitivities.append({"camera_type": kind, "sensitivity": label, **subset_stability(features, model, subset, config["seed"])})
            # Same cameras, same blocks, with the activation-selection weeks removed.
            later_features, _ = shape_features(first52.loc[features.index, 5:h], config)
            from model.camera_shape import fit_camera_shapes
            from sklearn.metrics import adjusted_rand_score
            later_model = fit_camera_shapes(later_features, model.n_clusters, config["seed"])
            sensitivities.append({"camera_type": kind, "sensitivity": "Omit first four weeks", "groups": len(later_features),
                                 "ari": float(adjusted_rand_score(model.predict(features.loc[later_features.index]), later_model.labels_)), "status": "Refit with weeks 5–52"})
    trends = pd.concat(results, ignore_index=True).rename(columns={"time": "week"})
    calendar_trends = pd.concat(calendars, ignore_index=True).rename(columns={"time": "week_start"})
    member = pd.concat(memberships, ignore_index=True) if memberships else pd.DataFrame(columns=["camera_id", "cluster_id", "cluster_label"])
    membership = eligibility.reset_index(drop=True).merge(member, how="left", on="camera_id", validate="one_to_one")
    membership["sparse_history"] = membership.nonzero_weeks.le(config["sparse_max_nonzero_weeks"]).astype("boolean").where(membership.nonzero_weeks.notna())
    profile = pd.concat(profiles, ignore_index=True) if profiles else pd.DataFrame()
    curve = pd.concat(cluster_curves, ignore_index=True) if cluster_curves else pd.DataFrame()
    diag = pd.concat(diagnostics, ignore_index=True)
    coverage = eligibility.groupby(["camera_type", "detection_confidence"]).agg(groups=("camera_id", "size"), eligible_26w=("eligible_26w", "sum"), eligible_52w=("eligible_52w", "sum"), eligible_104w=("eligible_104w", "sum"), eligible_administrative_52w=("eligible_administrative_52w", "sum")).reset_index()
    checks = validate_panels(meta, weekly, primary, administrative, eligibility, membership, config)
    assert checks.passed.all(), checks.loc[~checks.passed].to_string()
    tables = {"shared_trends": trends, "calendar_trends": calendar_trends, "camera_eligibility": eligibility.reset_index(drop=True),
              "cluster_membership": membership, "cluster_profiles": profile, "cluster_curves": curve,
              "cluster_diagnostics": diag, "cluster_sensitivity": pd.DataFrame(sensitivities),
              "coverage": coverage, "validation_checks": checks}
    composition = membership[membership.cluster_id.notna()].groupby(["camera_type", "cluster_id", "cluster_label", "activation_year", "detection_confidence"]).size().reset_index(name="groups")
    tables["cluster_composition"] = composition
    examples = []
    for _, row in membership[membership.get("is_representative", pd.Series(False, index=membership.index)).eq(True)].iterrows():
        counts = primary.loc[row.camera_id, 1:h]
        base = counts.loc[baseline].mean()
        for week, value in counts.items():
            examples.append({"camera_id": row.camera_id, "search_query": row.search_query, "camera_type": row.camera_type, "cluster_id": row.cluster_id, "week": int(week), "tickets": float(value), "index": float(value/base*100) if base > 0 else None})
    tables["cluster_examples"] = pd.DataFrame(examples)
    for name, frame in tables.items():
        frame.to_csv(OUT / f"{name}.csv", index=False)
    provenance = {"generated_at": datetime.now(timezone.utc).isoformat(), "input_snapshot": "2026-10-14", "cutoff": config["cutoff"], "groups": len(meta), "weekly_rows": len(weekly), "configuration": config,
                  "models": model_summary, "inputs": {name: hashlib.sha256((HERE/"outputs"/name).read_bytes()).hexdigest() for name in ["camera_summary.csv", "camera_weekly_trends.parquet", "reconciliation.json", "registry_pin.json"]},
                  "definitions": {"unit": "Frozen normalized address plus camera type; not verified physical devices", "anchor": "First Monday on or after unchanged inferred activation; Week 1", "primary": "Only seven-day, source-complete weeks ending by the last actual ticket", "administrative": "Covered seven-day weeks through cutoff; no observed ticket after last record becomes zero, without assuming operation", "index": "100 times weekly count divided by own first 13 complete weeks' mean; omit zero or incomplete baseline", "uncertainty": "Pointwise 95% percentile bootstrap, 500 resamples of whole camera histories; not simultaneous bands or causal uncertainty", "clustering": "13 four-week means, log1p, within-camera z-score; K-means per camera type; retain sparse histories; exclude constant block profiles", "selection": "Highest silhouette passing silhouette>=0.20, mean subsample ARI>=0.75, smallest cluster>=max(10,ceil(0.02N)); k=2..6", "activation_caveat": "Activity-based activation may mechanically create an early decline. Original high_sustained labels can also reflect upstream duplicate-week aggregation; they are not gold-standard dates."}}
    (OUT / "analysis_manifest.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    dictionary = []
    meanings = {"week": "Relative calendar week; first full Monday-Sunday week is 1", "n": "Cameras contributing a valid raw count this week", "index_n": "Cameras with a complete positive normalization baseline and a valid count", "raw_mean": "Equal-camera arithmetic mean, unique tickets/camera/week", "index_mean": "Mean of within-camera indices; own baseline=100", "raw_median": "Median count across cameras", "index_median": "Median within-camera index", "raw_q25": "25th percentile of camera counts", "raw_q75": "75th percentile of camera counts", "index_q25": "25th percentile of indexed camera counts", "index_q75": "75th percentile of indexed camera counts", "raw_ci_low": "2.5th percentile of whole-camera bootstrap means", "raw_ci_high": "97.5th percentile of whole-camera bootstrap means", "index_ci_low": "2.5th percentile of bootstrap index means", "index_ci_high": "97.5th percentile of bootstrap index means", "raw_mean_4w": "Trailing four consecutive weekly mean counts", "index_mean_4w": "Trailing four consecutive mean indices", "silhouette": "Within/between-cluster separation; higher is better", "stability_ari": "Mean adjusted Rand agreement over 12 fits on 80% subsamples, predicting all original groups", "distance_to_centroid": "Euclidean distance in 13 standardized log-block features", "shape_mean": "Cluster centroid in standardized log-block units", "sparse_history": "At most 13 positive weeks in the first 52 weeks; retained in model", "median_change_pct": "Median of camera-specific last-13/first-13 mean changes; zero baselines omitted", "confidence": "Unchanged Eric activation confidence label; not a cluster probability", "baseline_mean": "Mean of anchored weeks 1-13, only when all are complete", "is_representative": "One of three members closest to its fitted centroid"}
    for table, frame in tables.items():
        for column in frame.columns:
            dictionary.append({"table": table, "field": column, "meaning": meanings.get(column, column.replace('_', ' ').capitalize() + "; see analysis_manifest.json and notebook methods"), "dtype": str(frame[column].dtype)})
    pd.DataFrame(dictionary).to_csv(OUT / "field_dictionary.csv", index=False)
    install_browser_queries(tables, provenance)
    print(json.dumps({"coverage": rows_json(coverage), "models": model_summary, "checks_passed": int(checks.passed.sum())}, indent=2), flush=True)
    return tables, provenance


def install_browser_queries(tables, provenance):
    path = HERE / "browser/src/data.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    source = {"name": "Frozen camera histories: shared trends and shape clusters", "type": "file", "provider": "Local analysis of the October 14, 2026 snapshot", "executedAt": provenance["generated_at"],
              "files": ["camera_summary.csv", "camera_weekly_trends.parquet", "shared_trends/analysis_manifest.json"],
              "metricDefinitions": [{"label": k.replace('_', ' ').capitalize(), "definition": v} for k, v in provenance["definitions"].items()],
              "evidenceFlow": [{"title": "Frozen inputs", "detail": json.dumps(provenance["inputs"])}, {"title": "Configuration", "detail": json.dumps(provenance["configuration"])}, {"title": "Validation", "detail": "All panel reconciliation checks passed; raw weekly inputs and inferred activation dates are unchanged."}]}
    for name, frame in tables.items():
        snapshot["queries"][f"analysis_{name}"] = {"rows": rows_json(frame), "source": source}
    snapshot["queries"]["analysis_models"] = {"rows": provenance["models"], "source": source}
    snapshot["buildStatus"] = "updating"
    snapshot["generatedAt"] = provenance["generated_at"]
    # The original app identity and all individual-camera queries remain intact.
    path.write_text(json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"), allow_nan=False), encoding="utf-8")


def build_notebook():
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    existing = pd.read_csv(OUT / "shared_trends.csv") if (OUT / "shared_trends.csv").exists() else pd.DataFrame()
    findings = ""
    if not existing.empty:
        for kind, result in existing[existing.scenario.eq("balanced_52")].groupby("camera_type"):
            start = result[result.week.le(13)].raw_mean.mean()
            end = result[result.week.gt(39)].raw_mean.mean()
            index = result[result.week.gt(39)].index_mean.mean()
            findings += f"- **{kind}, n={int(result.n.iloc[0]):,}:** mean weekly count changes from {start:.1f} in weeks 1–13 to {end:.1f} in weeks 40–52. The final-window mean normalized index is {index:.1f} (each camera's first-13-week baseline = 100).\n"
        findings += "\nThe 52-week red-light cohort contains only 46 of 617 original groups. Its increasing mean index coexists with declining and increasing individual shapes; it should not be generalized to all red-light locations. The speed-camera average declines, but K=2–6 does not pass the preset separation requirement. Neither finding establishes a camera effect.\n"
    cells = [md("# Shared Ticket Trends and Camera Shape Clusters\n\nYihan Hu · Frozen issue-date window: January 1, 2016–August 30, 2026.\n\n## tl;dr\n\nThis notebook compares equal-camera ticket counts and normalized histories, then tests whether distinct temporal shapes are stable. All 4,610 original location/type groups remain in the eligibility and membership exports. The tables below report the actual eligible cohorts and selected models; excluded groups are not silently dropped."),
        md("## Context & Methods\n\nThe data are the frozen October 14 extract underlying the existing per-camera dashboard. No new downloads or activation estimates are introduced.\n\n### Key assumptions\n\n- Week 1 is the first full Monday–Sunday week on or after Eric's unchanged inferred activation. Missing early observations never shift the anchor.\n- Primary 26/52/104-week cohorts require every anchored week to be complete through the last actual ticket. These cohorts select longer observed histories and are not unbiased samples of all devices.\n- The fixed-cutoff sensitivity includes fully covered weeks through August 30, 2026, filling post-last-ticket weeks with zero observed tickets. It does not establish continued camera operation.\n- Counts are equally weighted by enforcement group. The normalized index divides each camera by its first 13-week mean; zero/incomplete baselines are excluded only from indexed results. Weeks 5–17 replace that baseline in the omit-first-four sensitivity.\n- Confidence intervals are pointwise 95% camera-history bootstrap intervals (500 resamples); the interquartile range describes heterogeneity instead. Spatial/common-calendar dependence is not modeled in those intervals.\n- Clustering uses 13 four-week means, log1p and within-camera standardization. Constant block profiles are excluded; sparse histories are retained and checked separately. K=2..6 is evaluated using separation, stability and cluster size.\n- Activation is inferred from ticket activity, creating selection and regression-to-the-mean concerns. Original confidence labels are not validation probabilities; upstream duplicate-week aggregation can affect their interpretation. Counts do not establish violation rates, behavior change, or causal safety effects."),
        md("### 1. Reproduce the analysis from frozen aggregates"),
        code("from pathlib import Path\nimport sys\nimport pandas as pd\nfrom IPython.display import display\nROOT = Path.cwd()\nwhile not (ROOT / 'pyproject.toml').exists():\n    if ROOT.parent == ROOT: raise RuntimeError('Run inside the repository')\n    ROOT = ROOT.parent\nDELIVERY = ROOT / 'experiments/camera-ticket-trends'\nsys.path.insert(0, str(ROOT))\nsys.path.insert(0, str(DELIVERY))\nfrom build_shared_analysis import run_analysis, OUT\nfrom shared_presentation import plot_eligibility, plot_shared, plot_sensitivities, plot_calendar, plot_clusters, plot_diagnostics, plot_cohorts\nfrom contextlib import redirect_stdout\nimport io\nwith redirect_stdout(io.StringIO()):\n    tables, manifest = run_analysis()\ndisplay(pd.DataFrame(manifest['models']).round(3))"),
        md("## Data\n\n### 2. Coverage and eligibility\n\nCompleteness of downloaded files does not imply complete early-2016 history. The same groups contribute throughout each fixed window."),
        code("display(tables['coverage'])\ndisplay(tables['camera_eligibility'].groupby(['camera_type','exclusion_reason']).size().rename('Groups').reset_index())\nplot_eligibility(tables, OUT)"),
        md("## Results\n\n### 3. Shared trajectories\n\nMean counts answer how many tickets a location records; normalized indices answer how a location changes relative to its own baseline. A large mean–median gap indicates concentration or heterogeneity."),
        code("plot_shared(tables, OUT)"),
        code("primary = tables['shared_trends'].query(\"scenario == 'balanced_52'\")\ncomparison = []\nfor kind, frame in primary.groupby('camera_type'):\n    first, last = frame[frame.week.le(13)], frame[frame.week.gt(39)]\n    comparison.append({'Type':kind,'Raw cohort':int(frame.n.iloc[0]),'Indexed cohort':int(frame.index_n.iloc[0]),'First 13-week mean':first.raw_mean.mean(),'Last 13-week mean':last.raw_mean.mean(),'Last 13-week average index':last.index_mean.mean()})\ndisplay(pd.DataFrame(comparison).round(2))"),
        md("### 4. Endpoint, confidence and activation-window sensitivities\n\nThese comparisons change the cohort or estimand, so disagreement is informative. The omit-first-four curve uses a weeks 5–17 baseline; it is not a continuation on the original index scale."),
        code("plot_sensitivities(tables, OUT)"),
        md("### 5. Calendar-time and activation cohorts\n\nThe calendar view has changing membership; its contributing-camera count is shown explicitly. Activation-year comparisons remain descriptive and can differ in geography, traffic, policy, operation and ascertainment."),
        code("plot_calendar(tables, OUT)\nplot_cohorts(tables, OUT)"),
        md("### 6. Shape clusters and model selection\n\nCluster labels describe the fitted trajectory. They do not identify causal mechanisms or verified operational status. A stable partition is not proof of natural categories."),
        code("display(tables['cluster_diagnostics'].round(3))\nplot_diagnostics(tables, OUT)\ndisplay(tables['cluster_profiles'].round(2))\nplot_clusters(tables, OUT)"),
        md("### 7. Cluster robustness and representative locations\n\nAdjusted Rand index compares membership agreement, with 1 meaning identical assignments up to label names. Representatives are actual members nearest each centroid."),
        code("display(tables['cluster_sensitivity'].round(3))\nrepresentatives = tables['cluster_membership'].query('is_representative == True')\ndisplay(representatives[['camera_id','search_query','camera_type','cluster_label','weekly_mean_52w','detection_confidence']].round(2))\ndisplay(tables['cluster_composition'].head(16))"),
        md("### 8. Reconciliation and provenance"),
        code("display(tables['validation_checks'])\nassert tables['validation_checks'].passed.all()\ndisplay(pd.DataFrame([{'Input':k,'SHA-256':v} for k,v in manifest['inputs'].items()]))"),
        md("## Takeaways\n\nRead the mean, median, indexed trend and sensitivity views together. Clustering describes variation around the shared trend; an average can conceal opposite trajectories. All exclusions and original confidence labels are retained in `camera_eligibility.csv` and `cluster_membership.csv`. The accompanying dashboard provides the same aggregate tables and links cluster members back to their full original ticket histories. No result establishes reduced speeding, confirmed camera operation, or an effect on collisions.")]
    if findings:
        cells.insert(1, md("### Observed results\n\n" + findings))
    nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})
    path = HERE / "Shared_Ticket_Trends_and_Clusters.ipynb"
    NotebookClient(nb, timeout=1800, resources={"metadata": {"path": str(ROOT)}}, kernel_name="python3").execute()
    nbformat.validate(nb)
    nbformat.write(nb, path)
    html, _ = HTMLExporter().from_notebook_node(nb)
    path.with_suffix(".html").write_text(html, encoding="utf-8")
    print(f"Executed {path.name}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-only", action="store_true")
    args = parser.parse_args()
    if args.analysis_only:
        run_analysis()
    else:
        build_notebook()
