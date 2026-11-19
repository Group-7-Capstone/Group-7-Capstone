# Current Status (as of Nov 16 2026)

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

DOT won't provide the MIT "NYC Walks" pedestrian model, but City Form Lab publishes it ([project page](https://cityform.mit.edu/projects/nycwalks)): GeoJSON or SHP with peak-period pedestrian volume per sidewalk and crosswalk segment, 2018–2019, no license stated. It's a one-time snapshot, so use it as a site feature in the baseline, not for changes over time. Combine it with the other exposure proxies: traffic counts (#35), MapPLUTO land use (#33), LION road class and lanes, and Citi Bike (#51). Following the paper, report results under several exposure versions rather than relying on one.
