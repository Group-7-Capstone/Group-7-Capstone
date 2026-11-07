# Group 7: Automated Traffic Enforcement in NYC

A three-month Columbia data science capstone with **NYC DOT Town+Gown**, under **Vision Zero**.

**Team 7:** Claire Gallagher, Eric Yi, Yihan Hu, Fei Xue, Jacob Boyar

## The problem
NYC runs about 1,200 red light cameras (RLCs). State law caps both the number of cameras and the program's cost, so every placement has to count. We aim to find the **site characteristics** (road geometry, traffic volume, environment) that make a camera **effective**, measured against a pre-installation baseline of speeds and severe crashes. The results will help DOT place future cameras on evidence, with attention to how cameras are distributed across neighborhoods.

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
| [LION street network](https://data.cityofnewyork.us/City-Government/LION/2v4z-66xt), release **26c** (assumed for all work so far) | `2v4z-66xt` |
| [Automated Traffic Volume Counts](https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt) | `7ym2-wayt` |
| [MapPLUTO](https://data.cityofnewyork.us/City-Government/Primary-Land-Use-Tax-Lot-Output-Map-MapPLUTO-/f888-ni5f) | `f888-ni5f` |
| [DOF Parking Violation Codes](https://data.cityofnewyork.us/Transportation/DOF-Parking-Violation-Codes/ncbg-6agr) | `ncbg-6agr` |
| Parking Violations Issued (one dataset per fiscal year), e.g. [FY2027](https://data.cityofnewyork.us/City-Government/Parking-Violations-Issued-Fiscal-Year-2027/pvqr-7yc4) | `pvqr-7yc4` |

## KSI crash model: shared assumptions
Every KSI model uses the defaults in `preprocessing/config.py` and `validation/validate.py`:
- **Sites:** LION intersections, plus one mid-block site per street segment for crashes more than 100 ft from an intersection.
- **Outcome:** a KSI proxy, using only years under the post-2019 injury coding. Features cover the 3 years before each prediction year.
- **Test:** a pooled holdout. The model trains on prediction year 2022 and is scored on each site's total KSI for 2023–25.
- **Features:** the standard sets in `preprocessing.panel.FEATURE_SETS`.
- **Cameras:** `data/camera_data/camera_activation_dates.csv`. Cameras label sites but aren't model features.

**Known limitations**
- **LION version:** one 2026 snapshot (26c) describes 2017–2025 crashes. Road changes made after a crash, such as bike lanes or road diets that are often added because of crashes, leak into the features and can inflate their effect. Future models should match each prediction year to the LION release in effect at that time.
- **Speed camera locations:** cameras are geocoded to "street @ cross street," so most mid-block speed cameras land on intersection sites (#45).

The weekly camera-effect model (`eric/`), the STGCN graph model (`jacob-branch`) and the ticket trends (`yihan/`) answer different questions and don't follow these defaults.

**Pending:** MIT "NYC Walks" pedestrian model ([City Form Lab](https://cityform.mit.edu/projects/nycwalks)) and DOT crash narratives (MV-104).

## Contributing1
Work is tracked in GitHub Issues grouped into epics. Keep PRs to one issue and under about 500 lines. Rules for coding agents and contributors are in [`AGENTS.md`](AGENTS.md), and current work is in [`CURRENT_STATUS.md`](CURRENT_STATUS.md).
