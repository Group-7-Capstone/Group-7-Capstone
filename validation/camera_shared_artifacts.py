"""Independent reconciliation of saved analysis, browser rows and notebook outputs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import nbformat

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / "experiments/camera-ticket-trends"


def validate():
    out = DELIVERY / "outputs/shared_trends"
    manifest = json.loads((out / "analysis_manifest.json").read_text(encoding="utf-8"))
    for name, sha in manifest["inputs"].items():
        assert hashlib.sha256((DELIVERY / "outputs" / name).read_bytes()).hexdigest() == sha
    summary = pd.read_csv(DELIVERY / "outputs/camera_summary.csv").set_index("camera_id")
    weekly = pd.read_parquet(DELIVERY / "outputs/camera_weekly_trends.parquet")
    eligibility = pd.read_csv(out / "camera_eligibility.csv").set_index("camera_id")
    trends = pd.read_csv(out / "shared_trends.csv")
    snapshot = json.loads((DELIVERY / "browser/src/data.json").read_text(encoding="utf-8"))
    assert len(eligibility) == len(summary) == 4610
    assert eligibility.inferred_activation_date.equals(summary.loc[eligibility.index].inferred_activation_date)
    primary = trends[trends.scenario.eq("balanced_52")]
    counts = weekly[weekly.is_complete_week].set_index(["camera_id", "week_start"]).observed_ticket_count
    reconciled = 0
    for kind, results in primary.groupby("camera_type"):
        eligible = eligibility[eligibility.camera_type.eq(kind) & eligibility.eligible_52w]
        matrix = []
        for camera_id, row in eligible.iterrows():
            activation = pd.Timestamp(row.inferred_activation_date)
            anchor = activation + pd.Timedelta(days=(-activation.weekday()) % 7)
            vector = [counts.loc[(camera_id, anchor + pd.Timedelta(weeks=k))] for k in range(52)]
            matrix.append(vector)
        matrix = np.asarray(matrix, dtype=float)
        index = matrix / matrix[:, :13].mean(axis=1, keepdims=True) * 100
        results = results.sort_values("week")
        assert (results.n == len(matrix)).all()
        assert np.allclose(results.raw_mean, matrix.mean(axis=0))
        assert np.allclose(results.raw_median, np.median(matrix, axis=0))
        assert np.allclose(results.index_mean, index.mean(axis=0))
        assert np.allclose(results.raw_q25, np.quantile(matrix, .25, axis=0))
        assert (results.raw_ci_high >= results.raw_ci_low).all()
        assert np.isclose(index[:, :13].mean(), 100)
        reconciled += matrix.size
    browser_checks = 0
    for query_id, query in snapshot["queries"].items():
        if not query_id.startswith("analysis_") or query_id == "analysis_models":
            continue
        table = pd.read_csv(out / f"{query_id[9:]}.csv")
        browser = pd.DataFrame(query["rows"])
        assert len(table) == len(browser), query_id
        for column in table.select_dtypes(include="number").columns:
            assert np.allclose(table[column], pd.to_numeric(browser[column]), equal_nan=True), (query_id, column)
        for column in ["camera_id", "cluster_id", "camera_type", "scenario", "inferred_activation_date"]:
            if column in table:
                assert table[column].fillna("").tolist() == browser[column].fillna("").tolist(), (query_id, column)
        browser_checks += len(table)
    members = pd.read_csv(out / "cluster_membership.csv")
    profiles = pd.read_csv(out / "cluster_profiles.csv")
    assert len(members) == 4610 and members.camera_id.is_unique
    assert members.loc[~members.eligible_52w, "cluster_id"].isna().all()
    for row in profiles.itertuples():
        assert members.cluster_id.eq(row.cluster_id).sum() == row.groups
        assert set(row.representative_ids.split(";")).issubset(set(members.loc[members.cluster_id.eq(row.cluster_id), "camera_id"]))
    nb = nbformat.read(DELIVERY / "Shared_Ticket_Trends_and_Clusters.ipynb", as_version=4)
    nbformat.validate(nb)
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert [c.execution_count for c in code] == list(range(1, len(code)+1))
    assert not [o for c in code for o in c.outputs if o.output_type == "error"]
    figures = sum("image/png" in o.get("data", {}) for c in code for o in c.outputs)
    assert figures == 7
    result = {"input_hashes_verified": len(manifest["inputs"]), "original_groups_retained": len(members), "independently_reconciled_primary_camera_weeks": reconciled,
              "browser_table_rows_checked": browser_checks, "notebook_code_cells": len(code), "embedded_figures": figures, "notebook_error_outputs": 0}
    (out / "artifact_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    validate()
