# Next steps after the Q2 risk map

*Branch `eric-roadway-features`, November 12, 2026.*

## Where things stand

| Work | Code | Notebook / write-up |
|---|---|---|
| Road attributes (Q2) | `preprocessing/roadway.py` | `q2_roadway_attributes.ipynb`, `q2_roadway_attributes.md` |
| Traffic volume (Q2, counted sites) | `preprocessing/traffic_volume.py` | section 3 of `q2_roadway_attributes.ipynb` |
| Design-only risk map | `preprocessing/intersections.py` | `q2_design_risk_map.ipynb` |
| Camera directions and locations | `preprocessing/camera_directions.py`, `camera_locations.py` | `data/camera_data/camera_directions.csv` |

### Q2 findings so far (details in `q2_roadway_attributes.md`)
- **Crash history ranks best.** It ranks next year's KSI sites best. Adding road attributes doesn't change the top-1,000 capture (15.8% both ways), because past crashes already reflect a site's design.
- **Road design still explains risk.** A 4-way intersection has 2.7× the KSI of a mid-block site, with crash history held equal. Truck routes, bike lanes, more lanes and wider streets are riskier; one-way streets are slightly safer.
- **It isn't just traffic.** On the 3% of sites with DOT traffic counts, adding volume leaves the design effects essentially unchanged. Only the truck route effect weakens.

## How the risk map works

**The problem:** Q1 and Q2 rank *crash sites*, which are built from crash locations. Places that never had a crash aren't in the data, and crash history can't flag a place without a record.

**The approach:**
1. **Units:** every LION intersection, meaning every node where 3+ streets meet. Nodes within 50 ft of each other are merged, since a divided road meets a cross street at several nodes. That gives 41,897 intersections, crashed or not.
2. **Crash assignment:** each crash goes to the nearest intersection within 100 ft. Intersections with no crashes keep zeros, so the model sees safe places too. The panel steps are the same as Q1's: yearly counts, 3-year lags, 2022–2025.
3. **Design model:** a negative binomial model learns how expected KSI depends on road attributes alone (legs, lanes, width, lane width, speed limit, one-way, highway, bike lane, truck route). It then scores every intersection, so one that looks like the dangerous ones scores high even with no crashes. Road-safety engineers call this a *safety performance function*.
4. **Testing:** the same expanding-window validation as Q1: fit on earlier years and score 2024 and 2025. Every model is also scored only on intersections with **no KSI** in the prior 3 years, and only on those with **no crash at all**.

## Risk map findings

| Intersections scored | Crash history (Q1 model) | Design only | Random picks |
|---|---|---|---|
| All intersections | 18.7% | 9.9% | 2.4% |
| No KSI in prior 3 years | 17.6% | 10.7% | 2.7% |
| No crash at all in prior 3 years | 8.9% | **21.3%** | 8.9% |

*Share of next year's KSI crashes at each model's top 1,000 intersections, averaged over 2024 and 2025.*

- **Most severe crashes happen where there was no recent severe crash.**
  - 85% of intersections had no KSI crash in 2022–2025.
  - 63% of KSI crashes happened at an intersection with no KSI in the 3 years before: 60% where there were minor crashes, and 3% where there was no crash at all.
- **Overall, crash history still ranks best.** Design alone captures about half as much, which is still about 4× random. Combining them doesn't improve the ranking (18.4%).
- **With no crash at all, design is the only signal, and it works.** Crash history has nothing to go on there, so it does no better than random. Design does 2.4× better than random. Few KSI crashes happen at such places (about 85 a year), so the gain is about 10 more KSI crashes a year found in the top 1,000.
- **With minor crashes but no KSI, the minor crashes are the better warning.** Crash history does 17.6%, design 10.7%.
- **What makes an intersection risky by design,** compared with a 3-way intersection where noted:

  | Attribute | Multiplier on expected KSI |
  |---|---|
  | Truck route | 2.0× |
  | 5+ legs (vs. 3-way) | 2.8× |
  | 4 legs (vs. 3-way) | 2.3× |
  | Bike lane (likely more cyclists) | 1.6× |
  | Each extra travel lane | 1.25× |
  | Posted 40+ mph (few pedestrians) | 0.5× |
- **The map:** 334 of the 1,000 riskiest intersections by design had no KSI crash in 2019–2025.
  - Many are on wide arterials (Queens Boulevard, Merrick Boulevard, Francis Lewis Boulevard) and have plenty of minor crashes.
  - Parts of Queens Boulevard were redesigned by DOT in recent years, which may explain their drop in severe crashes.
  - Some are highway crossings, likely false positives (next step 1).
- **Cautions:**
  - LION is a 2026 snapshot, so a recent redesign is treated as always there.
  - There's no traffic volume citywide.
  - These are associations, not causes.
  - Only two test years, and the no-crash group has few KSI crashes, so its numbers are noisy.

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
