# Group 7 Capstone: NYC DOT Red Light Camera Effectiveness

A three-month Columbia data science capstone with the **NYC DOT enforcement unit**, under **Vision Zero**.

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
| `eda/`, `experiments/` | Numbered notebooks that import from the layers |
| `tests/` | `pytest` tests |

## Setup
The team uses [uv](https://docs.astral.sh/uv/) on both macOS and Windows.
```bash
uv sync              # create or update .venv from uv.lock
uv run pytest        # run the tests
uv run jupyter lab   # notebooks: select the .venv kernel
```
To add a package, run `uv add <pkg>`. Dev tools go in the dev group: `uv add --dev <pkg>`.

## Data sources (NYC Open Data)
| Dataset | ID |
|---|---|
| [Motor Vehicle Collisions – Crashes](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Crashes/h9gi-nx95) | `h9gi-nx95` |
| [Motor Vehicle Collisions – Person](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Person/f55k-p6yu) | `f55k-p6yu` |
| [Motor Vehicle Collisions – Vehicles](https://data.cityofnewyork.us/Public-Safety/Motor-Vehicle-Collisions-Vehicles/bm4k-52h4) | `bm4k-52h4` |
| [LION street network](https://data.cityofnewyork.us/City-Government/LION/2v4z-66xt) | `2v4z-66xt` |
| [Automated Traffic Volume Counts](https://data.cityofnewyork.us/Transportation/Automated-Traffic-Volume-Counts/7ym2-wayt) | `7ym2-wayt` |
| [MapPLUTO](https://data.cityofnewyork.us/City-Government/Primary-Land-Use-Tax-Lot-Output-Map-MapPLUTO-/f888-ni5f) | `f888-ni5f` |
| [DOF Parking Violation Codes](https://data.cityofnewyork.us/Transportation/DOF-Parking-Violation-Codes/ncbg-6agr) | `ncbg-6agr` |
| Parking Violations Issued (one dataset per fiscal year), e.g. [FY2027](https://data.cityofnewyork.us/City-Government/Parking-Violations-Issued-Fiscal-Year-2027/pvqr-7yc4) | `pvqr-7yc4` |

**Pending:** MIT "NYC Walks" pedestrian model ([City Form Lab](https://cityform.mit.edu/projects/nycwalks)) and DOT crash narratives (MV-104).

## Background
- **Wilmington, DE:** [program report](https://deldot.gov/Programs/red_light/pdfs/WilmRedLightCameraProgramReportwithAppendices.pdf), [site selection methodology](https://deldot.gov/Programs/red_light/pdfs/2015ERLSPSiteSelectionReportwithAppendices.pdf)
- **Bellevue, WA:** [Vision Zero StoryMap](https://storymaps.arcgis.com/stories/a955ef85c8f245eaabbda05742e6167f), [methodology](https://bellevue.legistar.com/View.ashx?M=F&ID=14991801&GUID=0EA20F5F-FF00-47A0-B57D-2C619D8F5BE9)
- **Chicago:** [equitable RLC distribution](https://www.chicago.gov/content/dam/city/depts/cdot/Red%20Light%20Cameras/2022/Sutton+Tilahun_Chicago-Camera-Ticket_Exec%20Summary-Final-Jan10.pdf)
- **Camera deactivation and relocation effects:** [Springer article](https://link.springer.com/article/10.1007/s43762-022-00043-0)
- Also reviewed: *Causal decision-making for speed camera allocation* and *Can speed cameras make streets safer? Quasi-experimental evidence from New York City*

## Contributing
Work is tracked in GitHub Issues grouped into epics. Keep PRs to one issue and under about 500 lines. Rules for coding agents and contributors are in [`AGENTS.md`](AGENTS.md), and current work is in [`CURRENT_STATUS.md`](CURRENT_STATUS.md).
