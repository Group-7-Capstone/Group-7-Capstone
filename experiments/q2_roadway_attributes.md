# Q2: Road design and KSI risk, plus camera directions

*Branch `eric-roadway-features`, updated November 11, 2026. Notebook: `experiments/q2_roadway_attributes.ipynb`.*

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

> **Correction (November 12, 2026):** LION stores some street segments on more than one row, and the first version counted each row as an intersection leg. That inflated leg counts at about 19,000 LION nodes: many 4-way intersections were counted as 5+, and some street bends were counted as intersections. The tables below use corrected counts. The conclusions are unchanged, and the intersection effects got stronger.

### 1. Road attributes barely change the ranking

| Model | Top 100 | Top 1,000 | Deviance vs. Q1 |
|---|---|---|---|
| Crash history only (Q1) | 2.6% | 15.8% | n/a (baseline) |
| + intersection | 2.7% | 15.8% | −1.7% |
| + size | 2.3% | 15.7% | −0.7% |
| + speed limit | 2.7% | 15.8% | −0.1% |
| + road type | 2.8% | 15.5% | −0.5% |
| + all road attributes | 2.4% | 15.8% | **−2.7%** |
| Road attributes only (no crash history) | 0.9% | 7.6% | +8.0% |

The overall fit improves a little, but the top-100 list barely moves. A site's crash history already reflects its design. Top-100 capture varies by chance between models (2.3–2.8%), so differences under about one point aren't meaningful.

For scale: the 89,073 sites are ranked out of the whole city. Picking 100 sites at random would capture about 0.11% of KSI crashes, so the model is about 23× better than chance at the top 100 and 14× at the top 1,000.

### 2. Road design does explain risk

Multipliers on expected KSI crashes, with 95% intervals clustered by site:

| Attribute | Road only | With crash history |
|---|---|---|
| 3-leg intersection (vs. mid-block) | 3.03 | 2.28 |
| 4-leg intersection | 6.23 | 2.71 |
| 5+ leg intersection (0.4% of sites) | 5.68 | 2.39 |
| Truck route | 1.56 | 1.15 |
| Bike lane | 1.47 | 1.22 |
| Travel lanes (per lane) | 1.26 | 1.05 |
| Street width (per 10 ft) | 1.08 | 1.07 |
| Lane width (per ft) | 1.05 | 1.03 |
| One-way street | 0.87 | 0.93 |
| Highway or ramp | 1.43 | 1.04 (n.s.) |
| Speed limit 30–35 (vs. 25) | 0.72 | 0.96 (n.s.) |
| Speed limit ≤ 20 or 40+ | n.s. | n.s. |

n.s. = not statistically significant.

- **Intersections are the strongest factor.** With crash history held equal, a 4-way intersection has 2.7× the KSI of a mid-block site.
- **Truck routes and bike lanes most likely reflect exposure** (heavy traffic, more cyclists), not the infrastructure itself.
- **Wider roads and wider lanes are modestly riskier,** which fits wider lanes encouraging faster driving.
- **One-way streets are slightly safer,** with fewer conflicting turns.
- **Speed limit has no clear effect,** because almost every street is 25 mph.

### 3. Road design alone finds a lot of the risk

With no crash history at all, road attributes capture 7.6% of next year's KSI crashes in the top 1,000, about half of what the full model does. This is the basis for flagging dangerous designs before a severe crash happens.

### 4. The design effects aren't just traffic volume

DOT's Automated Traffic Volume Counts (`7ym2-wayt`, 2014 onward) give a typical daily volume for 1,368 counted LION segments (`preprocessing/traffic_volume.py`). Each site gets the volume of its busiest counted street within 100 ft, which covers **2,694 sites (3%) holding 8.6% of KSI crashes**. DOT counts busy streets, so KSI per site-year there is about 2.8× the citywide average. Everything below uses only these sites.

| Model (counted sites) | Top 100 | Top 250 | Deviance |
|---|---|---|---|
| Crash history (Q1) | 21.8% | 42.7% | 0.327 |
| + volume | 21.8% | 42.7% | 0.328 |
| + all road attributes | 20.4% | 43.0% | 0.326 |
| + road attributes + volume | 20.6% | 43.7% | 0.327 |
| Volume only | 4.6% | 12.8% | 0.426 |
| Road attributes only | 11.7% | 28.5% | 0.368 |
| Road attributes + volume | 10.3% | 29.5% | 0.369 |

| Multiplier (counted sites, no crash history) | Road only | Road + volume |
|---|---|---|
| 4-leg intersection | 8.05 | 8.18 |
| Street width (per 10 ft) | 1.14 | 1.14 |
| Travel lanes (per lane) | 1.08 (n.s.) | 1.05 (n.s.) |
| One-way street | 0.71 | 0.70 |
| Bike lane | 1.37 | 1.37 |
| Truck route | 1.23 | 1.15 (n.s.) |
| Daily volume (per doubling) | 1.23 alone | 1.11 |

- **Volume on its own predicts KSI.** KSI per site-year rises from 0.043 at the quietest fifth of counted sites to 0.131 at the busiest, about 3×.
- **Once road design is known, volume adds little.** Its multiplier drops from 1.23 to 1.11 per doubling, because lanes, width and truck routes already track volume (correlations around 0.5). With crash history in the model, volume adds nothing to the ranking.
- **The design effects don't shrink when volume is added.** The intersection, width, one-way and bike lane effects are unchanged; only the truck route effect weakens, consistent with it partly meaning "heavy traffic". So the design effects are not just "busy road", as far as one street's count can tell.
- **Limits:**
  - **One count is noisy.** For 129 segments DOT counted in two studies, log volume correlates only 0.58 between studies, which understates volume's true effect.
  - **Only the busiest street is measured.** An intersection's cross-street traffic is missing.
  - **Counted streets are DOT's choice,** not a random sample.

### Cautions

- **Traffic volume only covers 3% of sites** (section 4). For the other 97%, part of the road-size effects may still be "more cars."
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

1. **Better traffic volume (done for counted sites, section 4).** Coverage and noise are the limits now. Options:
   - Extend each count to the rest of the same street.
   - Add cross-street volume at intersections.
   - Use a modeled citywide volume (e.g. a DOT or MPO travel-demand estimate) so every site has one.
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
