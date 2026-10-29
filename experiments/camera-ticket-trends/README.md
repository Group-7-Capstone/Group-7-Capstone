# Per-Camera Ticket Trends Based on Eric's Results

This deliverable follows all 4,610 original normalized-address/camera-type groups from Eric's unchanged inferred activation dates to the last actual ticket observed on or before August 30, 2026. All authored content uses English. The notebooks, shared-trend and clustering analysis, LION features, and standalone dashboard are shared on `yihan/pipeline-integration`. The dashboard runs locally after download; website hosting is not included.

## Shared pipeline integration

The active experiment now lives under `experiments/camera-ticket-trends/`. The branch `yihan/pipeline-integration` is based on the team main branch containing PR #22. The original worktree and its analytical outputs remain the rollback reference. This upload uses a separate branch and does not merge into main or open a pull request.

Raw reads use `data.reader`: the team's `read_table(..., columns=...)` and `read_camera_data("camera_activation_dates")` are retained alongside bounded ticket batches and the LION reader. The current registry is checked against the original recorded content hash without invoking Git at analysis runtime. `registry_pin.json` retains historical provenance; `registry_reader_receipt.json` records the current shared-reader check.

The camera download command delegates to `data.download.fetch_soda`. It verifies the existing snapshot **offline by default**. `--download` explicitly permits fetching missing pages; use a new cache directory for a different extract. Completed frozen snapshots are not refreshed in place. Optional `columns`, `snapshot_dir`, `partition_field`, `fiscal_year`, `page_size`, and `offline` arguments support the camera workflow without changing the original three-argument team API. Page checksums, cursor chains, counts, exact queries and source-update checks are retained.

Set `CAMERA_TICKET_CACHE` in a local, ignored `.env` file to reuse a frozen cache outside this worktree, or pass `--cache`. No user-specific path is embedded in tracked code. Preprocessing also accepts `--output-dir` and `--work-dir`; its intermediate partitions stay separate from experiment outputs.

Reusable LION matching is available independently of this dashboard:

```python
from data.reader import read_lion_road_attributes
from preprocessing.camera_road_features import match_camera_roads, ROAD_CONFIG

roads, aliases, receipt = read_lion_road_attributes(lion_path)
features, candidate_segments = match_camera_roads(camera_summary, roads, aliases, ROAD_CONFIG)
```

`camera_summary` supplies the existing identifiers, address/type groups, borough labels, coordinates, activation dates and confidence labels. The returned tables retain one row per input group and the candidate-segment audit. Input tables are not modified. This is first-named-street context for camera groups, not verified enforced approaches. Matching all crash-derived sites and adding predictors to Claire's model remain future work; the model and its site definitions are unchanged.

Numerical transformation, model fitting and uncertainty/QA functions remain in their respective shared layers. Experiment scripts and notebooks orchestrate those functions and format results. The frozen ticket window, cohorts, normalization, clustering criteria and road categories are unchanged.

### Integration acceptance

The merged repository suite passed **74 tests**. Rebuilding from the frozen raw ticket pages passed **41,500 reconciliation checks**, and rebuilding the LION analysis passed **1,735 checks**. Comparison with the original workspace covered **30 analytical tables (1,000,436 rows)**: integers and categories match exactly, and the maximum floating-point difference is **0.0** (allowed tolerance: `rtol=1e-10`, `atol=1e-8`).

All three notebooks ran from clean kernels: 14, 9 and 19 code cells respectively, with no error outputs. All **22 embedded figures** are byte-identical to the reference notebooks. The four dashboard pages, camera search, type/confidence/coverage filters, relative/calendar axis, cluster selection, unknown-road toggle, CSV exports and chart PNG export were checked through localhost. Exported weekly, shared and road-category data reconcile to the saved tables.

The original workspace's 491 delivery files, 414 snapshot files and 58 LION files retain their original hashes; its Git status is unchanged. Evidence is retained in `outputs/integration/`, including the original baseline, result comparisons, source-preservation checks and browser review. `outputs/deliverable_manifest.json` records current delivery checksums separately from the original baseline. Automatic date and compact-number formatting in shared browser components follows the browser locale; authored text and static figures remain English.

This integration does not add LION attributes to all research sites or retrain the team's risk model. The reusable matching function is ready for that subsequent work. Local result files and caches remain ignored by Git.

## Open the results

Download [Camera_Ticket_Trends.html](Camera_Ticket_Trends.html) using GitHub's **Download raw file** control, then open the downloaded file in a browser. The approximately 56 MiB file embeds the reviewed aggregate snapshot and all four tabs: **Individual cameras**, **Shared trends**, **Camera clusters**, and **Road characteristics**. GitHub displays or downloads HTML source; it does not run this dashboard as a hosted website. Viewing the standalone file requires neither Python nor the raw datasets. Its historical local-only wording describes the original analysis run; this branch now shares the saved artifact.

The three executed notebooks and their HTML reading copies are included below. Their saved outputs can be read without running cells. Re-execution requires the repository's pinned environment, frozen ticket cache and supplied LION snapshot; those inputs and the separate `outputs/` tables remain local and are not included in the branch.

[delivery_manifest.json](delivery_manifest.json) records the shared artifacts' SHA-256 hashes, saved notebook checks and validation summary. Artifact bytes are preserved across Windows and macOS checkouts.

- [Ticket_Trend_Processing_Walkthrough.ipynb](Ticket_Trend_Processing_Walkthrough.ipynb): step-by-step teaching notebook with real intermediate records, time alignment, complete-week eligibility, baseline arithmetic, equal-camera averages, smoothing, bootstrap uncertainty, and reconciliation to the dashboard.
- [Ticket_Trend_Processing_Walkthrough.html](Ticket_Trend_Processing_Walkthrough.html): reading copy of the walkthrough, with saved tables and figures.
- `Per_Camera_Ticket_Trends.ipynb`: executed notebook, including methods, coverage, reconciliation, and fixed low/median/high-volume examples for both camera types.
- `Per_Camera_Ticket_Trends.html`: static notebook rendering with saved outputs.
- `Camera_Ticket_Trends.html`: standalone interactive HTML export; open directly in a browser without a server.
- `browser/dist/index.html` (local build only): interactive directory, weekly curves, filters, and exports. Serve the **whole** `browser/dist` folder; its content-addressed JSON sidecar is required. This generated directory is not included in Git; use the standalone HTML after cloning.
- `outputs/camera_summary.csv`: all 4,610 groups with original attributes, counts, dates, eligibility, and quality flags.
- `outputs/camera_weekly_trends.parquet` and `.csv.gz`: unique group/week observations, including observed counts, coverage-aware counts, and four-week averages.
- `outputs/field_dictionary.csv`, `validation_checks.csv`, `reconciliation.json`, and `registry_pin.json`: definitions and audit trail.
- `outputs/figures/`: 300 dpi figures used by the notebook.
- `outputs/unmatched_addresses.csv`: unmatched normalized-address/type counts. Duplicate, missing-ID, and invalid-date audits remain local Parquet files when applicable.

The migrated preview is registered locally on port 4186 with Data's preview helper. It can restart when you sign in to Windows. It is not hosted on GitHub or accessible to teammates through your loopback URL. Open the existing preview before starting another server on the same port.

For a foreground, session-only server on another machine, run this from the experiment folder in PowerShell after installing uv:

```powershell
.\serve.ps1
```

Open the URL printed by the script: it serves the local build when available, or `http://127.0.0.1:4186/Camera_Ticket_Trends.html` from a fresh clone. Stop this foreground server with Ctrl+C. `serve.ps1 -Port 4188` selects another port; `-Uv <path-to-uv>` supports uv installations outside PATH. Search the directory by address or stable ID, select a row, and use **Time axis** to switch between calendar weeks and weeks since inferred activation. Use **CSV** for the selected group's weekly data and the shared chart actions menu to export a figure. All authored content is English; some automatic number/date labels in the shared browser components follow the browser's locale. Notebook and static PNG labels use English formatting.

The standalone HTML was exported through Data's verified offline-export tool. Direct `file://` opening could not be tested because the in-app browser blocks that protocol; the served localhost version was tested. No browser security settings were changed.

## Counting and coverage

The extraction includes codes 7 (red light) and 36 (speed), using actual issue dates from January 1, 2016 through August 30, 2026. The source set is exactly FY2017–FY2027, as requested. Requests, source metadata, pagination cursors, counts, and SHA-256 receipts are stored under `data/raw/camera_ticket_trends_20261014/`. No earlier heatmap-analysis cache is used.

**Complete downloads and complete historical coverage are different.** FY2016 is outside the requested source set. Weeks before July 4, 2016 therefore have unknown full-week coverage even when backdated tickets are present. The pipeline retains those tickets in `observed_ticket_count` and cumulative analysis totals but leaves coverage-aware `ticket_count` null. Missing coverage is never filled with zero. Source changes or incomplete source downloads also make coverage unavailable conservatively. Zero means no ticket was observed in covered data, not proof that equipment was operating.

The primary key is a stable hash of Eric's exact `search_query` and `camera_type`. The registry is pinned to Git commit `743e9c7e27f6a07a6567611406857b8443a2b3f4`; its canonical Git-blob SHA-256 is recorded in `registry_pin.json`. Coordinates, labels, dates, and all address-normalization quirks remain unchanged. His rule removes direction tokens and combines address columns; multiple directions or county labels are flagged without splitting or re-geocoding a group.

Tickets are deduplicated across fiscal-year files by `summons_number`, preferring the newest fiscal year and then descending SODA source row ID. All competing rows are retained in duplicate audits, with conflict and winner flags. Missing IDs and unparseable dates are excluded and audited. Normalization and matching follow Eric's original requirements, including both address columns being present. Each selected ticket contributes to at most one group and one week.

Calendar weeks start Monday. The first and last week are partial when the analysis interval does not cover all seven days. The four-week trailing mean requires four consecutive complete and covered weeks. The first/last comparison uses the first and final 13 weeks of the longest eligible run of at least 26 weeks; ties choose the earliest run. The two windows never overlap. Percentage change is null if the initial mean is zero. This comparison is descriptive and may reflect seasonality or other changes.

`analysis_ticket_count` counts matched unique tickets on/after unchanged activation, including any observed tickets in unavailable-coverage weeks. It reconciles to the sum of `observed_ticket_count`, not the sum of the coverage-filtered plotting field. `new_total_all_dates` additionally includes matched tickets before activation and is the quantity compared with Eric's old total. Eric's old `last_ticket_date` is a week-start date; `last_actual_ticket_date` is a daily ticket date. A fresh extraction can contain historical additions or revisions, so the old snapshot is not guaranteed to reproduce.

Inferred activation is not verified installation. The last observed ticket is not verified deactivation. A normalized group is not necessarily one physical camera. Counts are neither exposure-adjusted violation rates nor evidence of behavioral improvement or a causal safety effect.

## Reproduce the analysis

Work from the repository root with Python 3.13 and `uv`. `pyproject.toml`, `uv.lock`, and `.python-version` are the reproducible environment configuration. The local shared-trend extension adds scikit-learn through uv and records its resolved dependencies in the lockfile.

```powershell
uv sync --locked
uv run python -m data.camera_ticket_snapshot --workers 4
uv run python -m preprocessing.camera_ticket_trends
uv run python -m validation.camera_ticket_trends
uv run python experiments/camera-ticket-trends/build_deliverable.py
uv run python experiments/camera-ticket-trends/document_outputs.py
uv run python -m validation.camera_ticket_browser
uv run pytest --basetemp=.tools/pytest-camera-trends-local
```

The downloader is resumable and verifies cached pages against their receipts. It writes only the independent cache. The data-layer reader exposes bounded batches; preprocessing hashes identifiers into 64 disk partitions before deduplication and aggregation. The notebook reads only the finalized aggregate outputs. Raw ticket details are not sent to the browser.

The browser uses Data's shared chart, source-inspection, table, and export components. To rebuild it with the installed Data plugin, use its documented `scripts/data-app.mjs build --project-dir <browser> --separate-data` command and bundled Node runtime. A prepared build is included locally, so viewing it does not require the plugin. Set `buildStatus` in `browser/src/data.json` to `complete` after data/content changes are validated, then rebuild. Preserve the app ID.

The local raw caches, environment, generated browser bundle, and aggregate exports are excluded from Git. Keep the repository code, pinned environment files, registry Git commit, cached receipts/data, and this deliverable together for a fully offline rerun. The previous heatmap notebook remains untouched in the original checkout.

## Shared trends and clustering extension

The same dashboard now has **Individual cameras**, **Shared trends**, **Camera clusters**, and **Road characteristics** tabs. These extensions are included in this branch's standalone export. The original individual histories and notebooks are retained.

- `Shared_Ticket_Trends_and_Clusters.ipynb` and `.html`: executed companion analysis, uncertainty, sensitivity checks and cluster diagnostics.
- `outputs/shared_trends/`: shared and calendar-time means, medians, quartiles, pointwise 95% bootstrap intervals, all-camera eligibility and cluster membership, model diagnostics, sensitivity checks, profiles, representatives, field dictionary and input SHA-256 receipts.
- `outputs/shared_trends/figures/`: seven publication figures at 220 dpi.
- `build_shared_analysis.py`: reproducible orchestration and browser snapshot integration. It reads only the frozen aggregate inputs; no new ticket downloads are needed.
- `preprocessing/camera_shared_trends.py`, `model/camera_shape.py`, and `validation/camera_shared_trends.py` in the repository own transformations, model fitting, and validation respectively.

Week 1 is the first complete Monday–Sunday week on or after unchanged inferred activation. The fixed 52-week cohort includes **3,124 speed** and **46 red-light** groups. The latter represents only 46 of 617 original red-light groups; 371 have insufficient administrative follow-up, 157 lack complete early coverage, and 43 end too early or have another coverage gap. Conclusions apply to the eligible cohort.

Normalized histories use each camera's first 13-week mean as 100. Missing and zero baselines are handled separately. The fixed-cutoff sensitivity extends fully covered observations through August 30, 2026; zeros after the last ticket mean no observed tickets, not confirmed camera operation. Confidence-only, baseline-volume, activation-year, 26/104-week and omit-first-four-week comparisons are included.

Clustering uses four-week means, log1p and within-camera standardization, with K=2–6 evaluated separately by camera type. Selection requires silhouette ≥0.20, mean resampling ARI ≥0.75 and each group ≥max(10, 2% of modeled locations). The 46 eligible red-light groups support **two exploratory clusters: 24 declining and 22 increasing**. No speed solution meets all thresholds: the best candidate is stable (ARI about 0.976) but weakly separated (silhouette about 0.188). The directory explicitly retains those locations without accepted cluster assignments. Thresholds are analytical choices, not universal proof of natural categories.

To regenerate the extension from the repository root:

```powershell
uv sync --locked
uv run python experiments/camera-ticket-trends/build_shared_analysis.py
uv run pytest --basetemp=.tools/pytest-shared-trends-local
```

The notebook executes the analysis afresh from a clean kernel and saves its outputs. Rebuild `browser` with the installed Data plugin after that run; export a fresh standalone HTML from the verified build. Inputs remain local and receipt-backed. The original confidence labels concern inferred activation, not coordinate precision, clustering confidence or a statistical probability.

## Step-by-step trend walkthrough

The new walkthrough focuses on understanding the calculations rather than refitting camera clusters. It reruns the primary 52-week and calendar-time trends using the same production functions and frozen weekly records. Upstream ticket cleaning is documented from the original receipts; sensitivity curves are read from the existing saved analysis. It does not download raw tickets, change inferred activation dates, or overwrite the dashboard.

Run from the repository root:

```powershell
uv run python experiments/camera-ticket-trends/build_trend_walkthrough.py
```

The outputs include an executed notebook, HTML, eight figures, example baseline arithmetic, independently checked primary means, and saved-table/browser reconciliations under `outputs/trend_walkthrough/`. This folder alone is not a standalone rerun environment: retain the repository modules, pinned environment and frozen aggregate inputs. Reading the saved notebook or HTML does not require rerunning it. A served reading copy is available at `http://127.0.0.1:4186/Ticket_Trend_Processing_Walkthrough.html` while the local server is running; regenerating the walkthrough restores that copy after a dashboard rebuild.

## Shared trends by road characteristics

Open `http://127.0.0.1:4186/?view=1&tab=road-trends` while the local server is running. This extension preserves every original ticket query, camera ID, activation date, coordinate and cluster result. It adds three category comparison charts in a dedicated tab, with separate speed/red-light populations:

- **Travel lanes:** 1, 2, 3 and 4+ using LION `Number_Travel_Lanes`. Parking lanes are excluded. Two-way counts include both directions, not lanes per enforced approach.
- **Paved road width:** <10, 10–<15, 15–<20 and 20+ metres, using `StreetWidth_Min/Max` multiplied by 0.3048. These are fixed descriptive bins, not official engineering categories. Width intervals crossing a boundary remain unknown; no midpoint is substituted.
- **Traffic direction:** `T` is two-way; `A`/`W` are one-way. Divided roadbeds are unresolved rather than misclassified as one-way streets.

Road context refers to the **first named street**, not a verified enforced approach or the whole intersection. Matching requires the recorded borough, exact normalized street/alias text and a 60 m coordinate neighborhood. Intersection matches additionally require the cross street and a shared LION node with compatible level codes, with that node also within 60 m. Only ordinary `FeatureTyp=0`, `RB_Layer=B`, `RW_TYPE=1` records without an endpoint-level change are assigned. Multiple adjoining segments remain in the audit: a feature category is assigned only when all candidates agree. No arbitrary nearest-road tie break, coordinate repair or fuzzy street-name guess is made. Street-only matches use a weaker name/borough/coordinate check and enter curves only under the explicit sensitivity option; block-range endpoints are not verified.

All 4,610 groups remain in the match directory. The finalized match table has 2,034 intersection street matches and 316 street-only matches; a successful street match can still have an unknown lane/width/direction category. The primary 52-week speed cohort has 3,124 histories, of which 1,503 have resolved lane categories, 1,339 resolved width categories and 1,582 resolved direction categories. The 46-history red-light cohort has only 13, 11 and 16 resolved histories respectively. Unknown groups are retained in counts and optional curves, never treated as zero-valued attributes.

Means reuse the existing fixed-cohort method: first full week after inferred activation, complete 26/52/104-week histories, equal group weights, own first-13-week mean = 100. Zero baselines are excluded only from the index. Four-week smoothing leaves the first three points missing. The category detail chart shows the unsmoothed mean, median and pointwise 95% whole-history bootstrap bounds (500 resamples, seed 42); n<2 has no interval. Groups below 10 remain visible with a small-sample label. The baseline ≥5 tickets/week option checks denominator amplification. No clustering is refit.

These are descriptive associations with a supplied LION snapshot whose release has not been independently established. Road conditions at activation are not reconstructed. Bootstrap bounds do not account for shared intersections, spatial/calendar dependence, mapping uncertainty or historical road changes. Differences are not exposure-adjusted rates, driving-behavior improvements or causal camera/road effects. Current features and longer-history selection may bias historical comparisons. LION definitions: [NYC DCP Geosupport dictionary](https://nycplanning.github.io/Geosupport-UPG/appendices/appendix03/).

To reproduce from the repository root, pass the actual local source path:

```powershell
uv run python experiments/camera-ticket-trends/build_road_trends.py --lion-gdb <path-to-lion.gdb>
uv run pytest --basetemp=.tools/pytest-road-trends-local
```

The data-layer reader loads and hashes the supplied GDB; `preprocessing/camera_road_features.py` owns matching and categories; `validation/camera_road_features.py` independently reconciles group membership, denominators and weekly means. `build_road_trends.py` assembles the reviewed tables into the existing dashboard. `--reuse-matches` can reuse this extension's saved match table; use the full command after changing matching code or inputs. After validation, set `buildStatus` to `complete`, rebuild with the installed Data plugin and export a new standalone HTML as documented above.

`outputs/road_features/` contains:

- `camera_features.csv`: all original groups, road attributes, statuses and fixed-window eligibility.
- `camera_road_features.csv`: matching-only source for repeat aggregation.
- `matched_segments.csv`: every accepted candidate segment and original lane, width and direction values.
- `trends.csv`, `coverage.csv`, `memberships.csv`: all curves, denominators and explicit group membership for each analysis selection.
- `match_status.csv`: matching coverage by type and recorded borough.
- `field_dictionary.csv`, `analysis_manifest.json`, `lion_receipt.json`, `validation_checks.csv`: definitions, configuration, input checksums and reconciliation results.

CSV exports from the webpage follow the current selection. The match directory is filtered only by camera type, status and address search; follow-up/baseline controls apply to curves, not that all-original-groups directory. Raw data, separate output tables, caches and generated browser directories remain local and excluded from Git; reviewed notebooks and standalone HTML exports are included.
