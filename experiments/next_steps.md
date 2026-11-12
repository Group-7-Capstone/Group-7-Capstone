# Next steps after the Q2 risk map

*Branch `eric-roadway-features`, November 12, 2026.*

## Where things stand

| Work | Code | Notebook / write-up |
|---|---|---|
| Road attributes (Q2) | `preprocessing/roadway.py` | `q2_roadway_attributes.ipynb`, `q2_roadway_attributes.md` |
| Traffic volume (Q2, counted sites) | `preprocessing/traffic_volume.py` | section 3 of `q2_roadway_attributes.ipynb` |
| Design-only risk map | `preprocessing/intersections.py` | `q2_design_risk_map.ipynb` |
| Camera directions and locations | `preprocessing/camera_directions.py`, `camera_locations.py` | `data/camera_data/camera_directions.csv` |

Key results so far:
- Crash history ranks next year's KSI sites best, and adding road attributes or traffic volume doesn't improve that ranking.
- Road design still explains risk, and not just through traffic. A 4-way intersection has 2.7× the KSI of a mid-block site, with crash history held equal.
- At intersections with no crash at all in the prior 3 years, design alone finds 21% of their next-year KSI crashes in its top 1,000, against 9% for random picks.

## Setting up on another laptop

1. `git pull` on `eric-roadway-features`, then `uv sync`.
2. Copy over or re-download what git doesn't hold:
   - `data/lion/lion.gdb` (DCP LION, downloaded by hand)
   - `data/raw/` (the DuckDB cache: `crashes`, `persons`, and `tickets_fy2017`–`tickets_fy2027` for camera work)
3. Pull the traffic counts, if needed (about 2 minutes):
   ```
   uv run python -c "from data.download import fetch_soda; fetch_soda('7ym2-wayt', 'traffic_volume_counts', 'yr >= 2014')"
   ```
4. Run `uv run pytest`. All 56 tests should pass.

## Next steps, in priority order

### 1. Filter grade-separated crossings out of the risk map
Some of the top "risky by design, no KSI on record" intersections are highway crossings where the roads pass on different levels, such as Cross Bronx Expressway & Lawton Ave and Staten Island Expressway crossings. They have few or no crashes because traffic doesn't actually meet. Fixing this is needed before the list goes to anyone.
- **How:** LION records each segment end's vertical level, which should be the `NodeLevelF` / `NodeLevelT` fields (check the exact names in the LION data dictionary). Count a leg at a node only when the segment meets the node at the same level as the others, inside `find_intersections` in `preprocessing/roadway.py`.
- **Check:** after the change, highway crossings with zero crashes should drop out of the top-25 table in `q2_design_risk_map.ipynb`.
- **Note:** this changes Q2's intersection features too, so rerun both Q2 notebooks.

### 2. Combine design and crash history into one score (empirical Bayes)
This is the Highway Safety Manual method: each intersection's score is a weighted average of the design prediction and its own crash count.
- **Formula:** `score = w × predicted + (1 − w) × observed`, with `w = 1 / (1 + α × predicted × years)`, where `α` is the negative binomial dispersion from the design model.
- **Effect:** a place with a long record leans on its crashes, and a place with no record leans on design.
- **Where:** it fits and predicts, so it's a model in `model/count_models.py`, following the existing `fit`/`predict` interface. Validate it with `evaluate` like the others.
- **Test:** compare its top-1,000 capture with crash history alone, overall and on the no-record subsets (section 3 of the risk map notebook).

### 3. Export the scores for review
Write one row per intersection: streets, coordinates, design score, combined score and crash record. This lets the team and DOT review the top places in GIS or on Street View.
- **Decide with the team** where it lives. `data/output/` is gitignored; `data/camera_data/` is for camera datasets only.
- **Spot-check** 20–30 of the "risky but no KSI" intersections on Street View. Do they look dangerous?

### 4. Camera effectiveness by approach
At camera intersections, compare crashes on the enforced approach with those on the other approaches, before and after activation (difference-in-differences). The two groups share the same intersection, weather and neighborhood.
- **Data needed:** NYPD Motor Vehicle Collisions – Vehicles (check the dataset ID on NYC Open Data; we believe it's `bm4k-52h4`), which records each vehicle's `travel_direction`. Pull it with `fetch_soda` and a `crash_date >= '2017-01-01'` filter.
- **First check:** how often `travel_direction` is filled in. If it's mostly blank, this design won't work, so check before building anything.
- **Inputs we already have:** activation dates (`camera_activation_dates.csv`) and enforced directions (`camera_directions.csv`).
- **Outcome:** all injury crashes, with KSI as secondary, since KSI per approach is too rare.
- **Caveats to state:** drivers may slow on every approach (spillover), and cameras were placed on bad approaches (regression to the mean).

### 5. Rerun Q1 with the corrected camera locations
About 1 in 4 speed camera approaches sits more than 500 ft from the original geocode. Today `assign_cameras` in `preprocessing/panel.py` uses the original coordinates from `camera_activation_dates.csv`.
- **Change:** take `latitude`/`longitude` from `camera_directions.csv` (LION-based, joined on `search_query`).
- **Check:** do the Q1 camera flags and the "no camera" watch-list results change?

### Later
6. **Historical LION:** use each year's LION release so road attributes match the crash year.
7. **Better traffic volume:** extend each count along its street, and add cross-street volume.
8. **Non-linear model:** gradient boosting with a Poisson objective, explained with SHAP. A Vertex AI AutoML run could serve as a benchmark here.

## Housekeeping

- **Split the branch into PRs.** It's well over the ~500-line limit. Proposed split, one issue each (labeled `claude`):
  - **A. Road attributes and Q2:** `roadway.py` (with the leg fix), `read_lion`, `RoadwayConfig`, `poisson_deviance`, tests, and Q2 sections 0–2.
  - **B. Camera directions and locations:** `base.py`, `camera_directions.py`, `camera_locations.py`, `CameraDirectionConfig`, tests, and the camera CSVs.
  - **C. Traffic volume:** `traffic_volume.py`, `TrafficVolumeConfig`, tests, and Q2 section 3.
  - **D. Design-only risk map:** `intersections.py`, `IntersectionConfig`, tests, and `q2_design_risk_map.ipynb`.
- **Update `CURRENT_STATUS.md`.** It still describes Week 3.
- **Two `eric/` notebooks** (`roadway_geometry_and_network_complexity.ipynb`, `seasonal_and_weather_shocks.ipynb`) have an uncommitted, meaningless change to their Python version metadata. Discard it with `git checkout -- <file>`.
- **The notebooks are now edited directly.** The scripts that first generated them were temporary.
