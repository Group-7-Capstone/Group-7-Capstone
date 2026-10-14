# Per-Camera Ticket Trends Based on Eric's Results

Descriptive analysis for [issue #4](https://github.com/Group-7-Capstone/Group-7-Capstone/issues/4) follows all 4,610 original normalized-address/camera-type groups from Eric's unchanged inferred activation dates to the last actual ticket observed on or before August 30, 2026. All authored content uses English.

## Open the results

- [Executed notebook](Per_Camera_Ticket_Trends.ipynb): methods, saved tables, seven figures, reconciliation, and fixed low/median/high-volume examples for both camera types. Reading the saved results requires no source-data download.
- [Interactive dashboard file](Camera_Ticket_Trends.html): a standalone HTML export, approximately **32.65 MiB**, containing the interface and aggregate histories for all 4,610 groups. On GitHub, use **Download raw file**, then open the downloaded file in a browser. The repository file page is not a hosted dashboard.
- [Direct HTML download](https://github.com/Group-7-Capstone/Group-7-Capstone/raw/refs/heads/yihan/ticket-trend/yihan/notebook-deliverables/camera-ticket-trends/Camera_Ticket_Trends.html).

The HTML contains aggregate camera histories, not individual ticket details. It is a frozen result and does not update automatically. If your browser restricts opening a local HTML file, serve this folder on localhost. From the repository root:

```powershell
uv sync --locked
uv run --no-sync python -m http.server 4187 --bind 127.0.0.1 --directory yihan/notebook-deliverables/camera-ticket-trends
```

From this folder in PowerShell:

```powershell
.\serve.ps1
```

Then open **http://127.0.0.1:4187/Camera_Ticket_Trends.html**. Stop the local server with Ctrl+C. The launcher serves this uploaded HTML; the excluded `browser/` project is not needed. No public website hosting is configured by this upload. Search the directory by address or stable ID, select a row, and use **Time axis** to switch between calendar weeks and weeks since inferred activation. Use **CSV** for the selected group's weekly data and the shared chart actions menu to export a figure. Some automatic number/date labels in the shared browser components follow the browser's locale. Notebook and static PNG labels use English formatting.

The standalone HTML was exported through Data's verified offline-export tool. Direct `file://` opening could not be tested because the in-app browser blocks that protocol; the served localhost version was tested. No browser security settings were changed.

## Snapshot results

| Measure | Count |
|---|---:|
| Original enforcement groups | 4,610 |
| Speed / red-light groups | 3,993 / 617 |
| Source records | 45,880,884 |
| Unique summons numbers | 43,994,668 |
| Unmatched unique tickets | 1,754,648 |
| Matched tickets before inferred activation | 72,457 |
| Tickets in the analysis intervals | 42,167,563 |
| Group-week rows | 849,122 |
| Groups eligible for the 13-week comparison | 3,820 |
| Groups affected by incomplete early-2016 coverage | 326 |

All 11 requested fiscal-year files were downloaded and reconciled. This does **not** imply complete historical coverage for every group.

## Counting and coverage

The extraction includes codes 7 (red light) and 36 (speed), using actual issue dates from January 1, 2016 through August 30, 2026. The source set is exactly FY2017–FY2027, as requested. Requests, source metadata, pagination cursors, counts, and SHA-256 receipts are stored under `data/raw/camera_ticket_trends_20261014/`. No earlier heatmap-analysis cache is used.

**Complete downloads and complete historical coverage are different.** FY2016 is outside the requested source set. Weeks before July 4, 2016 therefore have unknown full-week coverage even when backdated tickets are present. The pipeline retains those tickets in `observed_ticket_count` and cumulative analysis totals but leaves coverage-aware `ticket_count` null. Missing coverage is never filled with zero. Source changes or incomplete source downloads also make coverage unavailable conservatively. Zero means no ticket was observed in covered data, not proof that equipment was operating.

The primary key is a stable hash of Eric's exact `search_query` and `camera_type`. The registry is pinned to Git commit `743e9c7e27f6a07a6567611406857b8443a2b3f4`; its canonical Git-blob SHA-256 is recorded in `registry_pin.json`. Coordinates, labels, dates, and all address-normalization quirks remain unchanged. His rule removes direction tokens and combines address columns; multiple directions or county labels are flagged without splitting or re-geocoding a group.

Tickets are deduplicated across fiscal-year files by `summons_number`, preferring the newest fiscal year and then descending SODA source row ID. All competing rows are retained in duplicate audits, with conflict and winner flags. Missing IDs and unparseable dates are excluded and audited. Normalization and matching follow Eric's original requirements, including both address columns being present. Each selected ticket contributes to at most one group and one week.

Calendar weeks start Monday. The first and last week are partial when the analysis interval does not cover all seven days. The four-week trailing mean requires four consecutive complete and covered weeks. The first/last comparison uses the first and final 13 weeks of the longest eligible run of at least 26 weeks; ties choose the earliest run. The two windows never overlap. Percentage change is null if the initial mean is zero. This comparison is descriptive and may reflect seasonality or other changes.

`analysis_ticket_count` counts matched unique tickets on/after unchanged activation, including any observed tickets in unavailable-coverage weeks. It reconciles to the sum of `observed_ticket_count`, not the sum of the coverage-filtered plotting field. `new_total_all_dates` additionally includes matched tickets before activation and is the quantity compared with Eric's old total. Eric's old `last_ticket_date` is a week-start date; `last_actual_ticket_date` is a daily ticket date. A fresh extraction can contain historical additions or revisions, so the old snapshot is not guaranteed to reproduce.

Inferred activation is not verified installation. The last observed ticket is not verified deactivation. A normalized group is not necessarily one physical camera. Counts are neither exposure-adjusted violation rates nor evidence of behavioral improvement or a causal safety effect.

## Reproduce the analysis

Use a full Git clone so the pinned registry commit is available. Work from the repository root with Python 3.13 and `uv`. `pyproject.toml`, `uv.lock`, and `.python-version` are the reproducible environment configuration. This implementation uses the existing pinned dependencies without changing the team's dependency files.

```powershell
uv sync --locked
uv run python -m data.camera_ticket_snapshot --workers 4
uv run python -m preprocessing.camera_ticket_trends
uv run python -m validation.camera_ticket_trends
uv run python yihan/notebook-deliverables/camera-ticket-trends/build_deliverable.py
uv run python yihan/notebook-deliverables/camera-ticket-trends/document_outputs.py
uv run pytest
```

The downloader is resumable and verifies cached pages against their receipts. It writes only the independent cache. The data-layer reader exposes bounded batches; preprocessing hashes identifiers into 64 disk partitions before deduplication and aggregation. The notebook reads only the finalized aggregate outputs. Raw ticket details are not sent to the browser.

The commands above regenerate the analysis and notebook. They do not recreate the standalone dashboard runtime. Rebuilding that HTML for a different snapshot additionally requires the original local shared-chart project and its export tooling. `validation.camera_ticket_browser` and `finalize_local.py` are optional helpers for that original project, not prerequisites for the notebook pipeline or for opening the uploaded HTML.

The local raw caches, environment, generated browser directory, and aggregate table exports are excluded from Git. The standalone HTML is included as an explicitly selected deliverable. A rerun creates local `outputs/` files: camera summaries, weekly Parquet and compressed CSV tables, a field dictionary, coverage reports, unmatched-address summaries, reconciliation checks, and 300 dpi figures. Keep the repository code, pinned environment, registry Git commit, and cached receipts/data together for a fully offline rerun. Downloading the source tickets again can produce revised results.

## Validation

The completed analysis passed **41,499 reconciliation checks** and the repository's **18 tests**. The notebook contains 14 sequentially executed code cells, seven embedded figures, and no error outputs. All 4,610 dashboard groups and 849,122 weekly payload rows were compared against the aggregate tables. Interactive review covers search, filters, sorting, axis switching, and CSV/PNG exports; it is representative UI testing, not individual manual review of every group.

SHA-256 of the archived notebook: `53d38ddb4d010f7d4d0b0f92888db23355cf1f85b624fa955bb669e93bfac134`.

SHA-256 of the standalone HTML: `05d424c79549f9094be938b74d6fda8de65c161ef07f8d613cbb1abf0ea640d4`.
