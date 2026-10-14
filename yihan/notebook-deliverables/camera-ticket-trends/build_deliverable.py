"""Build reviewed browser data and execute the portable analysis notebook."""
from pathlib import Path
import argparse
import json
import os
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from data.reader import read_eric_registry
from data.camera_ticket_snapshot import SOURCES
from preprocessing.camera_ticket_trends import attach_identity

OUT = HERE / "outputs"
TITLE = "Per-Camera Ticket Trends Based on Eric's Results"


def browser_snapshot(catalog_only=False):
    OUT.mkdir(exist_ok=True)
    original, pin = read_eric_registry()
    if catalog_only:
        summary = attach_identity(original).drop(columns=["activation", "camera_index"])
        summary["data_status"] = "Awaiting ticket snapshot"
        summary["analysis_ticket_count"] = None
        summary["quality_flags"] = ""
        series = {}
        audit = {"camera_groups":len(summary), "all_sources_complete":False}
    else:
        summary = pd.read_csv(OUT/"camera_summary.csv")
        weekly = pd.read_parquet(OUT/"camera_weekly_trends.parquet")
        series = {}
        for camera_id, frame in weekly.groupby("camera_id", sort=False):
            values = frame[["observed_ticket_count","ticket_count","moving_average_4w",
                            "is_complete_week","is_partial_week","source_complete"]]
            series[camera_id] = json.dumps({"start":str(frame.week_start.iloc[0].date()),
                "values":json.loads(values.to_json(orient="values",double_precision=6))}, separators=(",",":"))
        audit = json.loads((OUT/"reconciliation.json").read_text())
    rows = json.loads(summary.to_json(orient="records", date_format="iso", double_precision=15))
    source = {
        "name":"Eric's frozen registry and NYC Open Data camera tickets",
        "type":"file", "provider":"NYC Department of Finance / NYC Open Data",
        "executedAt":datetime.now(timezone.utc).isoformat(),
        "tables":[f"FY{fy}: {ds}" for fy,ds in SOURCES.items()],
        "files":["camera_summary.csv","camera_weekly_trends.parquet"],
        "links":[{"label":"Frozen Eric registry","url":pin["url"]}] + [
            {"label":f"FY{fy} tickets","url":f"https://data.cityofnewyork.us/d/{ds}"} for fy,ds in SOURCES.items()],
        "metricDefinitions":[
            {"label":"Weekly tickets","definition":"Unique summons numbers dated within the selected group's unchanged inferred activation date and its last actual ticket, capped at August 30, 2026."},
            {"label":"4-week mean","definition":"Mean across four consecutive complete, source-covered weeks. Partial or unavailable weeks break the average."},
            {"label":"First vs last 13 weeks","definition":"Means of non-overlapping 13-week windows in the longest complete run of at least 26 weeks. A zero baseline has no percentage change."}],
        "evidenceFlow":[{"title":"Registry","detail":f"Git {pin['commit']}; SHA-256 {pin['sha256']}; activation dates unchanged."},
            {"title":"Raw snapshot","detail":"Only codes 7 and 36 from FY2017–FY2027; issue dates January 1, 2016 through August 30, 2026. Requests and checksums saved in source receipts."},
            {"title":"Identity and counting","detail":"Newest fiscal year wins for duplicate summons numbers. Eric's exact address normalization is retained. Each camera/type/week appears once."},
            {"title":"Interpretation","detail":"Approximate enforcement groups, not verified devices. Last observed tickets are not deactivation dates. Counts are not traffic-adjusted rates or causal effects."}],
    }
    snapshot = {"surface":"dashboard","title":TITLE,"generatedAt":datetime.now(timezone.utc).isoformat(),
                "status":"observed", "buildStatus":"creating" if catalog_only else "updating", "filters":[],
                "queries":{
                    "camera_directory":{"rows":rows,"source":source},
                    "camera_series":{"rows":[{"camera_id":r["camera_id"],"weekly_payload":series.get(r["camera_id"],"")} for r in rows],
                                     "payloadColumns":["weekly_payload"],"source":source},
                    "snapshot_status":{"rows":[{k:v for k,v in audit.items() if isinstance(v,(int,float,bool,str))}],"source":source}}}
    target = OUT/"reviewed_snapshot.json"
    target.write_text(json.dumps(snapshot,ensure_ascii=False,separators=(",",":"),allow_nan=False),encoding="utf-8")
    browser_data = HERE/"browser/src/data.json"
    if browser_data.exists():
        old = json.loads(browser_data.read_text(encoding="utf-8"))
        for key in ["id","artifactId","theme","dashboardId"]:
            if key in old:
                snapshot[key] = old[key]
        browser_data.write_text(json.dumps(snapshot,ensure_ascii=False,separators=(",",":"),allow_nan=False),encoding="utf-8")
    print(f"Saved {len(rows):,} camera groups for browser.",flush=True)
    return snapshot


def build_notebook():
    audit = json.loads((OUT/"reconciliation.json").read_text())
    summary = pd.read_csv(OUT/"camera_summary.csv")
    validation = json.loads((OUT/"validation_summary.json").read_text())
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    cells = [md(f"# {TITLE}\n\nYihan Hu · Descriptive ticket histories · Based on Eric's unchanged registry"),
        md(f"## tl;dr\n\nAll **{len(summary):,}** original enforcement groups are retained. The new snapshot contains "
           f"**{audit['source_records']:,} raw camera-ticket records**; **{audit['analysis_ticket_count']:,} unique tickets** "
           "match these groups on or after their original inferred activation dates. "
           f"**{validation['passed']:,} reconciliation checks passed.**\n\n"
           f"Cross-file deduplication removes **{audit['duplicate_rows_removed']:,} repeated rows**. "
           f"**{audit['unmatched_records']:,} unique tickets** do not match this fixed cohort and are retained in the unmatched-address audit. "
           f"**{int((~summary.source_complete).sum()):,} groups** have incomplete early historical coverage; their missing weeks are not plotted as zeros.\n\n"
           "Open the companion browser to search every group. This notebook contains reproducible calculations, coverage, and deterministic low/median/high-volume examples. "
           "Inferred activation and last observed ticket dates do not establish installation or deactivation."),
        md("## Context & Methods\n\nThe cohort is frozen at Git commit `743e9c7e27f6a07a6567611406857b8443a2b3f4`. "
           "The observation window is January 1, 2016–August 30, 2026. Activation dates and grouping keys are preserved exactly. "
           "Raw inputs come only from the same FY2017–FY2027 datasets used by Claire and Eric. This fresh snapshot may include revisions and backdated tickets.\n\n"
           "### Key Assumptions\n\n"
           "- A camera means Eric's normalized address plus camera type; directions and some address distinctions are merged by his rule.\n"
           "- Codes 7 and 36 define the selected ticket population; no claim of verified physical cameras is made.\n"
           "- Duplicate summons numbers prefer the newest fiscal-year source, then descending source row ID. Invalid dates and missing identifiers are audited.\n"
           "- Zero means no observed ticket in a complete source snapshot, not proof of active equipment. Missing source coverage stays null.\n"
           "- The requested sources begin with FY2017. Weeks before July 4, 2016 lack assured complete coverage because FY2016 was not requested. Backdated tickets in those weeks remain in observed counts and analysis totals; plotted complete-snapshot counts are null.\n"
           "- Calendar weeks run Monday–Sunday. First/last partial weeks are shown but excluded from smoothing and comparisons.\n"
           "- A four-week mean needs four consecutive full weeks. First/last 13-week means use the longest complete run of at least 26 weeks, choosing the earliest run on ties.\n"
           "- Counts are not exposure-adjusted rates. Changes cannot be attributed to enforcement, compliance, or safety without additional evidence."),
        md("### 1. Load the frozen outputs"),
        code("from pathlib import Path\nimport sys, json\nimport pandas as pd\nfrom IPython.display import display\n"
             "ROOT = Path.cwd()\nwhile not (ROOT / 'pyproject.toml').exists():\n    ROOT = ROOT.parent\n"
             "DELIVERY = ROOT / 'yihan/notebook-deliverables/camera-ticket-trends'\n"
             "sys.path.insert(0, str(ROOT))\nsys.path.insert(0, str(DELIVERY))\n"
             "from presentation import plot_group, plot_overview, select_examples\n"
             "OUT = DELIVERY / 'outputs'\nsummary = pd.read_csv(OUT / 'camera_summary.csv')\n"
             "weekly = pd.read_parquet(OUT / 'camera_weekly_trends.parquet')\n"
             "audit = json.loads((OUT / 'reconciliation.json').read_text())\n"
             "display(pd.DataFrame([{'Groups':len(summary),'Weekly observations':len(weekly),'Tickets':int(summary.analysis_ticket_count.sum()),'Sources complete':audit['all_sources_complete']}]))"),
        md("## Data\n\nSources are immutable local receipt-backed extracts for this run. The following table reconciles every fiscal-year source."),
        code("sources = pd.DataFrame([{'Fiscal year':m['fiscal_year'],'Dataset':m['dataset_id'],'Expected records':sum(m['expected'].values()),'Downloaded records':sum(m['actual'].values()),'Complete':m['complete'],'Source changed during pull':m['source_changed_during_pull']} for m in audit['source_counts']])\ndisplay(sources)"),
        md("### 2. Verify group coverage and exclusions"),
        code("display(summary.groupby(['camera_type','detection_confidence','data_status'],dropna=False).agg(Groups=('camera_id','size'),Tickets=('analysis_ticket_count','sum')).reset_index())\n"
             "display(pd.DataFrame([{'Check':k.replace('_',' ').capitalize(),'Records':v} for k,v in audit.items() if isinstance(v,int) and not isinstance(v,bool)]))"),
        md("## Results\n\n### 3. Overall counts and observation coverage\n\nAggregate counts reflect a changing set of observed enforcement groups; they are not a balanced-panel estimate."),
        code("plot_overview(summary, weekly, OUT / 'figures')"),
        md("### 4. Individual group histories\n\nExamples are deterministic: the lowest, median-ranked, and highest positive ticket totals within each type, with camera ID breaking ties. The browser includes all groups, including empty histories."),
        code("examples = select_examples(summary)\ndisplay(examples[['camera_id','search_query','camera_type','inferred_activation_date','last_actual_ticket_date','analysis_ticket_count','quality_flags']])"),
    ]
    for i in range(6):
        cells += [md(f"### Example {i+1}"), code(f"row = examples.iloc[{i}]\nplot_group(row, weekly[weekly.camera_id.eq(row.camera_id)], OUT / 'figures')")]
    cells += [md("### 5. First versus last 13 complete weeks\n\nEach row compares its own eligible windows. Seasonal differences, policy changes, changing traffic, and uncertain operation can affect these descriptive differences."),
        code("eligible = summary[summary.first_13w_mean.notna()].copy()\ndisplay(eligible.groupby('camera_type').agg(Eligible_groups=('camera_id','size'),Median_absolute_change=('absolute_change','median'),Median_percentage_change=('percentage_change','median')).round(2))\n"
             "direction = eligible.assign(Change=eligible.absolute_change.map(lambda x: 'Higher final mean' if x > 0 else 'Lower final mean' if x < 0 else 'Unchanged mean'))\ndisplay(pd.crosstab(direction.camera_type, direction.Change))\n"
             "display(eligible[['camera_id','search_query','comparison_start','comparison_end','first_13w_mean','last_13w_mean','absolute_change','percentage_change']].head(10).round(2))"),
        md("### 6. Compare with Eric's saved observations\n\nThe comparison uses all matched dates within the extraction window, including tickets preceding inferred activation. It must not be compared directly with post-activation-only counts. Old last-ticket values are week starts; new values are actual ticket dates."),
        code("display(pd.DataFrame([{'Groups with different all-period totals':int(summary.count_difference_from_eric.ne(0).sum()),'Groups with changed last observed week':int(summary.last_week_changed.sum()),'Groups with quality flags':int(summary.quality_flags.fillna('').ne('').sum())}]))\n"
             "display(summary[['camera_id','total_tickets_issued','new_total_all_dates','count_difference_from_eric','last_ticket_date','last_actual_ticket_date']].head(10))"),
        md("## Takeaways\n\nUse the interactive directory to inspect the complete population before interpreting individual examples. A declining curve alone does not identify its cause. "
           "Eric's original activation estimates are retained even where low confidence or inherited normalization limitations apply. "
           "Any gap after the final observed ticket remains an observation limitation, not a confirmed shutdown.\n\n"
           "### Reproducibility\n\nFrom the repository root, run `uv sync --locked`, then the download, preprocessing, validation, and build commands in this folder's README. "
           "Reading saved outputs does not require a download. Raw caches are intentionally local."),
        code("checks = pd.read_csv(OUT / 'validation_checks.csv')\nassert checks.passed.all()\nassert len(summary) == 4610\nassert not weekly.duplicated(['camera_id','week_start']).any()\nprint(f'{len(checks):,} reconciliation checks passed; original activation dates preserved.')")]
    nb = nbformat.v4.new_notebook(cells=cells,metadata={"kernelspec":{"name":"camera-trends","display_name":"Camera Trends (uv)","language":"python"}})
    kernel_root = ROOT/".tools/jupyter"
    kernel = kernel_root/"kernels/camera-trends"
    kernel.mkdir(parents=True,exist_ok=True)
    (kernel/"kernel.json").write_text(json.dumps({"argv":[sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"],"display_name":"Camera Trends (uv)","language":"python"}))
    os.environ["JUPYTER_PATH"] = str(kernel_root)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT/".tools/jupyter-runtime")
    Path(os.environ["JUPYTER_RUNTIME_DIR"]).mkdir(exist_ok=True)
    path = HERE/"Per_Camera_Ticket_Trends.ipynb"
    nbformat.write(nb,path)
    NotebookClient(nb,timeout=1200,kernel_name="camera-trends",resources={"metadata":{"path":str(ROOT)}}).execute()
    nbformat.write(nb,path)
    html,_ = HTMLExporter().from_notebook_node(nb)
    (HERE/"Per_Camera_Ticket_Trends.html").write_text(html,encoding="utf-8")
    print(f"Executed {sum(c.cell_type=='code' for c in nb.cells)} code cells.",flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog-only",action="store_true")
    parser.add_argument("--snapshot-only",action="store_true")
    args = parser.parse_args()
    browser_snapshot(args.catalog_only)
    if not args.catalog_only and not args.snapshot_only:
        build_notebook()
