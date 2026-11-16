# Group 7: Automated Traffic Enforcement in NYC

A three-month Columbia data science capstone with **NYC DOT Town+Gown**, under **Vision Zero**.

**Team 7:** Claire Gallagher, Eric Yi, Yihan Hu, Fei Xue, Jacob Boyar

## The problem
NYC runs over 2,000 school-zone **speed cameras**. State law limits them to a fixed number of school zones (750 since July 2019), so every placement has to count. We aim to find the **site characteristics** (road geometry, traffic volume, environment) that make a camera **effective**, measured against an Empirical Bayes crash baseline from similar sites without cameras. We then compare placement rules: does ranking sites by **risk × camera effect** beat simply placing cameras where crashes are highest? The results will help DOT place future cameras on evidence, with attention to how cameras are distributed across neighborhoods.

**Why speed cameras, not red light cameras (agreed with DOT, Nov 2026):** about 3,000 speed cameras switched on between 2017 and 2024, which gives before and after periods to estimate an effect. Only a few dozen red light cameras did. Red light cameras can still go through the placement comparison using an effect taken from the literature.

We use **public data only**. DOT's internal camera data can't be released, so we reconstruct camera locations, activation dates and orientation from open data. DOT will cross-check any discrepancies we find.

## Pipeline
```
data/ → preprocessing/ → model/ → validation/
```
| Layer | Role |
|---|---|
| `data/` | Download **raw** data into a local database. `reader.py` returns DataFrames. |
| `preprocessing/` | All cleaning, filtering, joins, spatial snapping and features, plus the inferred camera attributes |
| `model/` | Fit models on preprocessed data |
| `validation/` | Standard metrics and data QA/QC for any model |
| `yourname/` | Place for R&D for each team member at the start |
| `tests/` | `pytest` tests |

### How the pieces fit
```
 DATA                       PREPROCESSING                MODELS                       EVALUATION
 crashes ──────────────┐
 LION roads ───────────┤
 NYC Walks pedestrians ┼──► site × year table ──┬──► A. Baseline ──────────────┐
 traffic counts ───────┘                        │    similar sites, no camera  │
                                                │    (SPF + Empirical Bayes)   ├──► C. Camera effect
 speed camera tickets ─┐                        │                              │    by site type
 DOT schedule ─────────┴──► activation dates ───┤                              │    (EB before–after,
                            + locations         │                              │     robustness checks)
 school locations ─────────► eligible sites ────┼──► comparison group ─────────┘            │
                                                │                                           ▼
                                                └──► B. KSI risk at candidates ──► D. Benefit = risk × effect
                                                                                            │
                                                                                            ▼
 Census / NTA ───────────────────────────────────────────────────────────────────► E. Placement rules
                                                                                      + stress test
                                                                                      + equity report
```
- **A. Baseline:** expected crashes at a site without a camera, from a model fit on comparable school-zone sites without cameras and combined with the site's own history (Empirical Bayes).
- **B. Risk:** expected KSI at candidate sites (the KSI model below).
- **C. Effect:** crashes after activation compared with A, grouped by site type. Checks: staggered rollout, enforced vs. unenforced hours, the August 2022 switch to 24/7, and several activation definitions.
- **D. Benefit:** B × C, the KSI a camera is expected to prevent at each eligible site.
- **E. Placement:** compare uniform, most-dangerous-first, status quo and risk × effect. The stress test shrinks how much C varies by site type, and the equity report shows who gets cameras under each rule. Census data is used only here, not as a model feature.

Which layer holds D and E is still open (see `CURRENT_STATUS.md`).

## Setup
The team uses [uv](https://docs.astral.sh/uv/) on both macOS and Windows.
```bash
uv sync              # create or update .venv from uv.lock
uv run pytest        # run the tests
uv run jupyter lab   # notebooks: select the .venv kernel
```
To add a package, run `uv add <pkg>`. Dev tools go in the dev group: `uv add --dev <pkg>`.

Optional: copy `.env.example` to `.env` and add your own free Socrata app token from the
[NYC Open Data portal](https://data.cityofnewyork.us/profile/app_tokens). The API works without one, but larger or repeated pulls are throttled harder unauthenticated.

## Data sources (NYC Open Data)
| Dataset | ID |
|---|---|
| [Motor Vehicle Collisions – Crashes](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95) | `h9gi-nx95` |
| [Motor Vehicle Collisions – Person](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Person/f55k-p6yu) | `f55k-p6yu` |
| [Motor Vehicle Collisions – Vehicles](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Vehicles/bm4k-52h4) | `bm4k-52h4` |
| [LION street network](https://www.nyc.gov/content/planning/pages/resources/datasets/lion): sites from release **26c**; road features year-matched (21d–25d, via `data.download.download_lion`) | `2v4z-66xt` |
| [Automated Traffic Volume Counts](https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt) | `7ym2-wayt` |
| [MapPLUTO](https://data.cityofnewyork.us/City-Government/Primary-Land-Use-Tax-Lot-Output-Map-MapPLUTO-/f888-ni5f) | `f888-ni5f` |
| [DOF Parking Violation Codes](https://data.cityofnewyork.us/Transportation/DOF-Parking-Violation-Codes/ncbg-6agr) | `ncbg-6agr` |
| Parking Violations Issued (one dataset per fiscal year), e.g. [FY2027](https://data.cityofnewyork.us/City-Government/Parking-Violations-Issued-Fiscal-Year-2027/pvqr-7yc4) | `pvqr-7yc4` |

**From DOT:** `data/camera_data/speed_camera_schedule.csv` is the speed-camera operating schedule (hours, days, months and permitted school zones for each period). Cameras ran 7am–5pm on school-year weekdays until June 2019, 6am–10pm on weekdays until July 2022, and 24/7 since August 1, 2022. Activation dates and effect estimates have to account for when cameras were allowed to issue tickets.

**Crash data start dates:** NYPD moved crash reports to an electronic system (FORMS) in March 2016, so person-level data starts in April 2016 and our crash data starts on January 1, 2017. The injury coding changed in 2019, so KSI-based features use only 2019 onward.

## KSI crash model: shared assumptions
Every KSI model uses the defaults in `preprocessing/config.py` and `validation/validate.py`:
- **Sites:** LION intersections, plus one mid-block site per street segment for crashes more than 100 ft from an intersection.
- **Outcome:** a KSI proxy, using only years under the post-2019 injury coding. Features cover the 3 years before each prediction year.
- **Test:** a pooled holdout. The model trains on prediction year 2022 and is scored on each site's total KSI for 2023–25.
- **Features:** the standard sets in `preprocessing.panel.FEATURE_SETS`. Road features come from the last LION release before each prediction year (`RoadwayConfig.release_by_year`, `roadway.load_roadway_features_by_year`).
- **Cameras:** `data/camera_data/camera_activation_dates.csv`. Cameras label sites but aren't model features.

**Known limitations**
- **LION version:** road features are year-matched, but site locations still come from 26c, and a single release per year can't show changes within that year. Using the 26c snapshot instead overstated protected bike lanes by about a third and pushed their estimated risk from 1.00 up to 1.10.
- **Speed camera locations:** cameras are geocoded to "street @ cross street," so most mid-block speed cameras land on intersection sites (#45).

The weekly camera-effect model (`eric/`), the STGCN graph model (`jacob-branch`) and the ticket trends (`yihan/`) answer different questions and don't follow these defaults.

**Pending:** DOT crash narratives (MV-104).

**Pedestrian exposure:** the MIT "NYC Walks" model ([City Form Lab](https://cityform.mit.edu/projects/nycwalks)) is publicly downloadable: estimated peak-period pedestrian volume on every sidewalk and crosswalk segment, 2018–2019. It is a single modeled snapshot, so it can describe how busy a site is but not changes over time.

## Contributing
Work is tracked in GitHub Issues grouped into epics. Keep PRs to one issue and under about 500 lines. Rules for coding agents and contributors are in [`AGENTS.md`](AGENTS.md), and current work is in [`CURRENT_STATUS.md`](CURRENT_STATUS.md).
