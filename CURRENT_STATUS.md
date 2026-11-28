# Current Status

> Temporary context. Update it as items are finished and delete what's resolved. The lasting rules are in `AGENTS.md`, and background is in `README.md`.

## Direction
DOT agreed (Nov 2026) that we focus on **speed cameras**. About 3,000 speed cameras switched on between 2017 and 2024, so we can measure their effect. Only a few dozen red light cameras did. The project now has two parts:
1. **Camera effect:** how much a speed camera cuts crashes, and at which kinds of sites.
2. **Placement:** does ranking sites by **risk × effect** beat placing cameras where crashes are highest? This idea comes from Nguyen (2026), "Learning to allocate automated speed enforcement" (*Case Studies on Transport Policy*). We keep its policy comparison and stress tests, at site level instead of borough level, and without its reinforcement learning.

## Plan
| Step | What | Status |
|---|---|---|
| Site × year table | LION intersection and mid-block sites, crash counts, road features, camera flags (`preprocessing/panel.py`) | Done |
| KSI risk model | `NegativeBinomialModel` with top-k and calibration validation (midterm notebook) | Done. Still to do: exclude or flag sites that already have a camera, so risk means "without a camera" |
| Speed-camera activation dates | `camera_activation_dates.csv` | Needs a speed-specific rule (see below) |
| Speed-camera locations | `camera_locations.py`, `camera_directions.py` | Needs a location check (see below) |
| Comparison group | Sites without cameras, minus a buffer around cameras (#27) | To do |
| Baseline | Model fit on comparison sites using site features only (no crash history), then Empirical Bayes expected crashes for camera sites with yearly adjustment factors | To do |
| Camera effect | EB before–after at camera sites, grouped by 3–4 site types chosen in advance. Checks: staggered rollout; enforced vs. unenforced hours; the August 2022 switch to 24/7 | To do |
| Placement comparison | Benefit = risk × effect. Compare uniform, most-dangerous-first, status quo and risk × effect, with a stress test that shrinks the differences in effect between site types | To do |
| Equity report | Who gets cameras under each rule, using Census data (#34) for evaluation only, not as model features | To do |

**Outcomes:** the KSI proxy is the main outcome, and all injury crashes are the robustness check. Before–after comparisons against a comparison group cancel the 2019 coding change, so the effect step can use 2017 onward. The risk model keeps KSI features from 2019 onward.

## Speed-camera data fixes (do these first)
- **Activation rule (#23), in review:** `preprocessing/cameras.py` replaces the `eric/` notebook (rebuild with `uv run python -m preprocessing.cameras`, about 10 minutes). Speed cameras are judged only on weeks the schedule allowed (no pre-2019 summers, no school holidays). 446 cameras running before the June 2016 data start are flagged `left_censored`: leave them out of before–after comparisons. 95 speed activations right after the July 2019 and August 2022 schedule changes get "medium" confidence. Tickets listed in two fiscal-year tables (1.9M) are counted once, and 1,017 locations with no sustained run and under 100 tickets (text typos) are dropped. The default threshold is **not** scaled by hours, because tickets per camera fall as hours grow (median 51 a week before 2019, 18 since 24/7). For stress tests, the file also has `activation_date_unadjusted` (no schedule), `activation_date_hours_scaled` and `first_ticket_date`; they move 32, 113 and 187 speed cameras to a different year.
- **Locations (#45):** 603 speed-camera rows share coordinates with another row, and some block-style locations are geocoded to the wrong borough (e.g. "1 AV E 7 ST – E 14 ST" lands in Brooklyn). Add a borough-consistency check.
- **Spillover:** speed cameras slow a whole corridor, so leave sites near a camera out of the comparison group (#27, #28).

## Things to verify
- **School zones:** we don't filter sites by school zone. A ¼-mile radius around schools covers most of the city, there's no public list of the 750 camera zones, and public school data is incomplete. Placement produces a ranked list that DOT screens for eligibility at the end. Still worth asking DOT how many of the 750 zones are in use and whether the cap changed after 2022, since that sets how many new cameras are realistic.
- **Weekly camera-effect model:** README mentions one in `eric/`, but it isn't on `main` or `eric-eda`. Ask Eric.
- **Red light camera activations:** 414 of 617 are dated 2025–26, which looks like a gap in ticket coverage rather than new installs. Confirm with DOT.

## Open decisions
- Which layer holds the benefit calculation and the placement comparison?
- Which 3–4 site types the effect may vary by. Fix these before looking at results.

## External data
- **Pending, don't build against yet:** DOT crash narratives (MV-104), which may be released later for QA/QC.

**NYC Walks pedestrian volume (#33), in review.**
- **What we use:** City Form Lab's ([project page](https://cityform.mit.edu/projects/nycwalks); Sevtsuk et al., *Nature Cities*, doi:10.1038/s44284-025-00383-y) published SHP of modeled peak-hour pedestrian volume per sidewalk and crosswalk segment for a typical 2018–19 weekday. No license is stated, so cite the paper. Download it with `data.download.download_nycwalks()`.
- **Features:** `preprocessing/pedestrians.py` takes the busiest segment within 100 ft of each site (weekday 8–9 AM + 12:30–1:30 PM + 5–6 PM) and gives `FEATURE_SETS["pedestrian"]`. 97% of intersections and 95% of mid-block sites match; the rest get the median and `ped_missing`.
- **Not used:** the paper's `.pckl` files are Random Forest calibrators. Rerunning them needs their `madina` network model and proprietary InfoGroup business data, so we use the published estimates only.
- **Caveats:** it's a pre-COVID snapshot, so it's a fixed site feature, not change over time. It underestimates the busiest places (held-out R² about 0.7 on weekdays).
- **Holdout check** (`experiments/pedestrian_exposure.ipynb`): **keep it in the EB baseline.**
  - Without crash history, top-1,000 KSI capture rises from 6.0% to 7.6% (+1.6 points, 95% interval +0.9 to +2.4), about 1.5× expected KSI per doubling of foot traffic.
  - With crash history the change is within noise (+0.2 points).
  - Speed-camera sites are busier on foot (standardized difference 0.27), but road features already explain that (about 0 after them).
- **Still to check:** injury and pedestrian-injury outcomes. `NegativeBinomialModel` doesn't converge on them, which needs a model-layer fix.
- **MapPLUTO (#33 part 2):** test it only as an addition to road-only + pedestrian.

Combine it with the other exposure proxies: LION road class and lanes, and Citi Bike (#51). Following the paper, report results under several exposure versions rather than relying on one.

**Traffic counts (#35), in review: don't use.** `preprocessing/traffic.py` turns DOT's Automated Traffic Volume Counts (`7ym2-wayt`, `data.download.download_traffic_counts()`) into weekday daily traffic per site, and `experiments/traffic_exposure.ipynb` tests it.
- Counts reach only 4.5% of sites (11.5% of KSI, 9.4% of speed-camera sites), mostly big roads. Filling in the rest from road features gives a cross-validated R² of 0.24.
- On the road-only + pedestrian model, no version (counted flag, imputed, counted-else-imputed, counted sites only) changes top-k KSI capture beyond noise. After road and pedestrian features, traffic is 1.05× KSI per doubling (0.95 to 1.17).
- Only a source that covers every segment (e.g. NYSDOT AADT) would be worth revisiting.
