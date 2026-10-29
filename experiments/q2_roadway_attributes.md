# Q2: Road design and KSI risk, plus camera directions

*Branch `eric-roadway-features`, October 29, 2026. Notebook: `experiments/q2_roadway_attributes.ipynb`.*

## The question

Q1 (`q1_ksi_risk.ipynb`) ranks crash sites by their expected KSI (killed or seriously injured) crashes next year, using crash history only. Q2 asks whether the road itself (intersection layout, lanes, width, speed limit, road type) explains KSI risk beyond that history, and whether it improves the ranking.

## Data and method

- **Road attributes** come from DCP's LION street network (`data/lion/lion.gdb`). Every LION street within 100 ft of a Q1 site describes the site (`preprocessing/roadway.py`):

  | Group | Attributes |
  |---|---|
  | Intersection | legs at the nearest intersection (3, 4, 5+, vs. mid-block), intersections within 100 ft |
  | Size | travel lanes and curb-to-curb width of the biggest street (both roadbeds of a divided road added together), estimated lane width |
  | Speed limit | posted limit: 20 or less, 30–35, 40+, vs. the 25 mph default |
  | Road type | one-way, highway or ramp, bike lane, truck route |

  99% of sites have a value for each attribute. Missing values are filled with the median and flagged.
- **Model and testing:** Q1's negative binomial model and validation, unchanged. The model is trained on earlier years and tested on 2024 and 2025: for 2024, it is fit on 2022–2023 and its predictions are compared with actual 2024 KSI crashes, then the same for 2025. A site's inputs are always its previous three years of crashes, and sites were defined using only crashes before 2024.
- **Metrics:**
  - **Top-k capture:** the share of next year's KSI crashes that happen at the model's top 100 or top 1,000 sites.
  - **Deviance:** how close predicted counts are to actual counts at every site; lower is better.

## Results

### 1. Road attributes barely change the ranking

| Model | Top 100 | Top 1,000 | Deviance vs. Q1 |
|---|---|---|---|
| Crash history only (Q1) | 2.6% | 15.8% | n/a (baseline) |
| + intersection | 2.8% | 16.0% | −1.6% |
| + size | 2.3% | 15.7% | −0.7% |
| + speed limit | 2.7% | 15.8% | −0.1% |
| + road type | 2.8% | 15.5% | −0.5% |
| + all road attributes | 2.5% | 15.9% | **−2.5%** |
| Road attributes only (no crash history) | 0.9% | 7.0% | +9.1% |

The overall fit improves a little, but the top-100 list barely moves. A site's crash history already reflects its design. Top-100 capture varies by chance between models (2.3–2.8%), so differences under about one point aren't meaningful.

For scale: the 89,073 sites are ranked out of the whole city. Picking 100 sites at random would capture about 0.11% of KSI crashes, so the model is about 23× better than chance at the top 100 and 14× at the top 1,000.

### 2. Road design does explain risk

Multipliers on expected KSI crashes, with 95% intervals clustered by site:

| Attribute | Road only | With crash history |
|---|---|---|
| 3-leg intersection (vs. mid-block) | 2.43 | 1.97 |
| 4-leg intersection | 5.06 | 2.40 |
| 5+ leg intersection | 5.63 | 2.62 |
| Truck route | 1.56 | 1.13 |
| Bike lane | 1.48 | 1.22 |
| Travel lanes (per lane) | 1.26 | 1.04 |
| Street width (per 10 ft) | 1.08 | 1.07 |
| Lane width (per ft) | 1.04 | 1.03 |
| One-way street | 0.89 | 0.92 |
| Highway or ramp | 1.25 | 1.00 (n.s.) |
| Speed limit 30–35 (vs. 25) | 0.69 | 0.95 (n.s.) |
| Speed limit ≤ 20 or 40+ | n.s. | n.s. |

n.s. = not statistically significant.

- **Intersections are the strongest factor.** With crash history held equal, a 4-way intersection has 2.4× the KSI of a mid-block site.
- **Truck routes and bike lanes most likely reflect exposure** (heavy traffic, more cyclists), not the infrastructure itself.
- **Wider roads and wider lanes are modestly riskier,** which fits wider lanes encouraging faster driving.
- **One-way streets are slightly safer,** with fewer conflicting turns.
- **Speed limit has no clear effect,** because almost every street is 25 mph.

### 3. Road design alone finds a lot of the risk

With no crash history at all, road attributes capture 7% of next year's KSI crashes in the top 1,000, about half of what the full model does. This is the basis for flagging dangerous designs before a severe crash happens.

### Cautions

- **No traffic volume.** Busy roads tend to be wide, multi-lane and truck routes, so part of these effects is just "more cars."
- **The intersection effect is partly how sites are defined.** A 100 ft circle at an intersection covers several approaches; mid-block it covers one stretch of street.
- **LION is a 2026 snapshot** applied to 2022–2025 crashes. Bike lanes and road diets are often added because of crashes, which can inflate their apparent effect.
- **Wide divided roads may be undercounted.** A 100 ft radius can catch only one of their two roadbeds.
- **Only two test years,** and calibration (are the predicted counts right, not just the ranking?) wasn't rerun for Q2.
- **These are associations, not causal effects.**

## Camera directions and locations

New code: `preprocessing/camera_directions.py` and `preprocessing/camera_locations.py`. Output: `data/camera_data/camera_directions.csv`, one row per enforced direction.

- **Direction of enforced traffic.** Every camera ticket records it in its location text (e.g. `ASTORIA BLVD S (E/B) @ 77TH ST`). Over 99.9% of camera tickets have a readable direction. Of the 4,610 camera locations, 338 enforce two directions.
- **Compass bearing.** Each direction is converted to a real heading using the camera's street in LION. A check: Manhattan avenue cameras marked "northbound" come out at a median of 29°, which is exactly the tilt of Manhattan's grid. 94% of approaches get a bearing.
- **Camera locations corrected.** The original geocodes in `camera_activation_dates.csv` misplace many cameras. Re-placing each camera at the LION intersection named in its own tickets moved 1 in 4 speed camera approaches more than 500 ft. Checked against the borough recorded on each location's tickets, borough agreement rose from 91.3% to 97.4% overall. For mid-block cameras that LION moved, it rose from 68% to 99%.
- **What "direction" means.** It is the direction of the ticketed traffic, not confirmed lens orientation. NYC cameras photograph from behind, but this should be confirmed with DOT before saying where a camera faces.
- **Consequence for Q1.** Q1 assigns cameras to sites using the original coordinates, so about a quarter of speed cameras may be attached to the wrong site.

## Next experiments

In rough priority order.

1. **Add traffic volume.** Join NYC DOT Automated Traffic Volume Counts (NYC Open Data) to sites, and rerun Q2 on the sites with counts. This is the only way to separate "badly designed" from "busy," so it comes first. Counts cover only some streets, so start with the subset of sites that have them.
2. **Rerun Q1 with the corrected camera locations.** Use the LION-based coordinates from `camera_directions.csv` in `assign_cameras` and check whether the camera flags and no-camera-site results change. This is a small change with a large data-quality payoff.
3. **Calibration and per-year results for Q2.** Use `calibration_table` and report 2024 and 2025 separately, to show whether predicted counts are right and how stable the gains are.
4. **Design-only risk map (all intersections).** Use every LION intersection as the unit, including ones with zero crashes, and fit a model on design (and volume) only. This finds risky places without a crash record, which crash-built sites can't.
5. **Camera effectiveness by approach.** At intersections where a camera enforces one direction, compare crash trends on the enforced vs. unenforced approaches before and after activation. Both approaches share the same intersection, weather and neighborhood, so this is a much fairer comparison than DOT's before/after numbers, which Q1 showed fall about two-thirds on their own at high-KSI sites. Add matched comparison sites from the Q1 risk scores.
6. **Non-linear model.** Gradient boosting with a Poisson objective, explained with SHAP, in the same validation. It tests interactions such as wide road × high speed and whether the negative binomial leaves accuracy on the table. A Vertex AI AutoML Tabular run is a cheap benchmark here; nothing else so far needs cloud compute.
7. **Historical LION.** Use LION releases from each year so road attributes match the crash year. This removes the snapshot leak and enables before/after studies of road redesigns (road diets, bike lanes).
8. **Validate by area as well as by year.** Hold out whole neighborhoods to check that results aren't driven by nearby sites sharing conditions.

## Open questions for the team

- **Commit the derived camera CSVs?** Should `camera_directions.csv` and the 2026 permanent-camera file with directions be committed for teammates without the raw tickets? AGENTS.md says not to commit data, but `data/camera_data/*.csv` is already shared.
- **Camera orientation:** confirm with DOT that cameras photograph the ticketed traffic from behind.
- **Q1 coordinates:** should Q1 switch to the LION-based camera locations (next experiment 2)?
- **PR split:** the branch holds two pieces of work, so it should become two PRs under about 500 lines each: (a) road attributes and Q2, (b) camera directions and locations.
