# Current Status (as of Week 3, Sept 28 2026)

> Temporary context. Update it as items are finished and delete what's resolved. The lasting rules are in `AGENTS.md`, and background is in `README.md`.

## Repo state
- The repo is a **skeleton**. All layer files (`base.py`, `reader.py`, `preprocessor_1/2.py`, `validate.py`) are empty, and `download.py` is a placeholder.
- The branches are `main` and `claire-eda`. `pipeline/pipeline_1.ipynb` is only an import test using `sys.path.append("..")`.
- **Still to add:** a `.gitignore` (covering data and database files such as `*.duckdb`, plus `__pycache__/`, `.DS_Store`, `.venv/`, `.env`, `.ipynb_checkpoints/`) and a `tests/` folder.
- **Switch to uv.** Every teammate installs uv. Then set up the project: `uv init --bare`, `uv python pin <version>`, `uv add pandas geopandas duckdb pyarrow requests`, and `uv add --dev pytest ipykernel ruff`. Commit `pyproject.toml`, `uv.lock` and `.python-version`. The existing `.venv` (Python 3.14) should be recreated with `uv sync`. Choose 3.12 or 3.14 depending on whether geopandas, pyproj and shapely have wheels for 3.14 on both Windows and Mac.

## Next steps
1. **Data layer:** load Crashes, Person and Vehicles raw into a local database, and implement `data/reader.py`. Use the bulk CSV export and do the download on your own laptop. Row counts since 2017, checked Sept 28: Crashes about **1.31M**, Person about **5.0M**, Vehicles a few million (the count request timed out). That's a few GB in total.
2. **Parking violations pull:** the FY2027 dataset (`pvqr-7yc4`) already has about **1.5M rows**, and the fiscal year only started in July. 2017 to the present across all annual datasets could be over 100M rows. Pull **only camera violation codes** (look them up in `ncbg-6agr` first) from each fiscal-year dataset through the SODA API. Log the code filter with the data.
3. **Preprocessing:** filter to fatal or severe crashes from 2017 onward (keep the cutoff as a config value), join on `collision_id`, and clean the fields.
4. **LION:** load it raw in the data layer and convert it to GeoPackage for spatial snapping.
5. **EDA:** look at spatial crash patterns relative to camera locations. This work goes in `eda/`.
6. **Infrastructure research:** look into GCP for large jobs, mirroring DOT's on-prem Postgres.

## In progress: reconstructing the camera dataset
Target table: `camera_id, lat, lon, LION node, approach, activation_date, method, confidence`.

| Step | Layer |
|---|---|
| Collect raw camera positions (e.g. from Waze alerts) and raw camera violation records | data/ |
| Infer activation dates from first-issuance changepoints | preprocessing/ |
| Work out each camera's orientation from ticket patterns | preprocessing/ |
| Reconcile with DOT's count of about 1,200 and flag ambiguous cameras for DOT to check | validation/ |

## Things to verify
- The violations dataset `pvqr-7yc4` is linked as "2026", but its page title says **FY2027**. Confirm which year it covers.
- Look up the exact red light and speed camera violation codes in `ncbg-6agr`.

## Open decisions
- **How do we define "effective"?** Options include change in fatal or severe crashes, violation trends, and right-angle crash reduction. Until this is decided, don't hard-code a definition.
- Which rule should link crashes to camera approaches: a buffer radius or the nearest LION node?
- How long should the before and after windows be, and how do we handle regression to the mean?
- Which local database? The recommendation is **DuckDB**: a single file with no server, fast CSV/Parquet reading, a spatial extension, and support on both Windows and Mac. Local Postgres is the alternative if mirroring DOT's setup matters. The team needs to confirm.

## Pending external data (don't build against these yet)
- **MIT "NYC Walks" pedestrian model:** Heidi Wolf at DOT is working on getting access.
- **DOT crash narratives (MV-104):** these may be released later for QA/QC.
