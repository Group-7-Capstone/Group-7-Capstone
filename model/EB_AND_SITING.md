# Empirical Bayes baseline and camera siting

How the Empirical Bayes (EB) baseline works, why it differs from the KSI risk model, and how its camera-effect estimates feed into siting. Steps A–E refer to the pipeline diagram in `README.md`. Open decisions are tracked in `CURRENT_STATUS.md`.

## Two models, two questions

| | KSI risk model (B) | EB baseline (A) |
|---|---|---|
| **Question** | Which sites will have KSI crashes next? | How many crashes would a camera site have had *without* the camera? |
| **Used for** | Ranking candidate sites by risk | Measuring the camera effect (C) |
| **Fit on** | All sites | Comparison sites only: no camera, and not near one (#27) |
| **Crash history** | Main input (3-year lagged counts) | Not a model input. It enters in the EB step instead (below) |
| **Judged by** | Holdout top-k capture and calibration | Whether it gives a fair counterfactual for camera sites |

The risk model uses whatever predicts best. The baseline has to stay unbiased for sites that got a camera, which constrains what it can use.

## How the EB baseline works

**Step 1: the safety performance function (SPF).** A negative binomial model of crash counts on site features only: road design, intersection type, block length and pedestrian volume. It is fit on comparison sites, with year effects. For a camera site it gives μ, the expected crashes per year at a typical site like it. The model's overdispersion α measures how much sites vary beyond what the features explain.

**Step 2: combine μ with the site's own record.** For a camera site with x crashes over its before-period of *n* years:

```
w           = 1 / (1 + α · μ · n)
EB expected = w · μ · n + (1 − w) · x
```

A site with a long or busy record (large μ·n) leans on its own count. A quiet site leans on μ. The estimate is then carried into the after-period with yearly adjustment factors (comparison-site trends), so citywide changes such as falling crash recording don't count as camera effects.

**Step 3: the camera effect.** For each camera site, compare after-period crashes with the EB expected count. The ratio observed ÷ expected is the crash modification factor (CMF): below 1 means fewer crashes than expected. The variance of each EB estimate comes from α, which is what gives effect estimates their intervals.

## Why crash history isn't in the SPF

Crash history does enter the EB method, but only in step 2, through x. Putting it in the SPF as well would break the method in three ways:

1. **Counting it twice.** The site's own before-period crashes would drive both μ and x, so a noisy count would get too much weight.
2. **Regression to the mean.** Cameras tend to go where crashes recently spiked. A model that treats a recent spike as lasting risk would carry it into the expected count. The after-period would then look safer than expected even if the camera did nothing. The site-features-only μ is the anchor that a spike shrinks toward, and that shrinkage is the correction EB exists to make.
3. **Contamination.** The expected count is projected through the after-period. With lagged crashes as inputs, after-period predictions would be built from crashes the camera already affected, so the counterfactual would contain part of the effect it measures.

A consequence: exposure has to come from site features. In the risk model, crash history already reflects how many people pass a site. The SPF has no such stand-in. That's why NYC Walks pedestrian volume matters much more for the baseline than for the risk model (`experiments/pedestrian_exposure.ipynb`). In the site-features-only model it raises top-1,000 KSI capture from 6.0% to 7.6%, about 1.5× KSI per doubling of foot traffic. With crash history it adds about +0.2 points, within noise.

## From camera effects to siting

**1. Estimate effects by site type, not by site.** A single site's ratio is very noisy, since most sites see only a few KSI crashes a year. Pool the ratios within 3–4 site types chosen in advance, giving a CMF and interval for each type. Candidate types include intersection vs. mid-block, arterial vs. local, and high vs. low foot traffic.
- **Pooled EB by type:** standard, transparent, few ways to overfit.
- **Weighted meta-regression:** the log ratio on site features, weighted by precision and shrunk toward the citywide effect. Richer, but more room to overfit.

Fix the types before looking at results. If the split is chosen after seeing effects, the "best" sites partly reflect which split happened to look good.

**2. Benefit = risk × (1 − CMF).** For each candidate site without a camera:

```
benefit = expected KSI per year without a camera (B) × (1 − CMF of the site's type) (C)
```

This is expected KSI prevented per year, and the ranking follows it. A busy mid-block arterial with a strong camera effect can outrank a riskier intersection where cameras do little. Risk must mean "without a camera", so sites that already have a camera are excluded or flagged in B.

**3. Compare placement rules (E).** Give each rule the same number of cameras and count expected KSI prevented under each:
- uniform placement
- most-dangerous-first (risk only)
- status quo (current camera sites)
- risk × effect

The **stress test** shrinks the differences between type CMFs step by step toward the pooled effect. If risk × effect still wins at half the heterogeneity, the result holds up. If its lead disappears, effect differences shouldn't drive siting yet. The equity report (#34) shows who gets cameras under each rule.

## Cautions

- **Only use effects where cameras exist.** If no cameras sit on a site type (say, quiet local streets), there's no effect to estimate there. Use the pooled effect for it, or leave it out. Check that camera and candidate sites overlap on the type-defining features.
- **Cameras weren't placed at random.** DOT chose sites under school-zone rules, so a type's effect describes cameras sited the way DOT sites them. That supports "more like these", less so very different sites.
- **Spillover and spacing.** Speed cameras slow whole corridors (#27, #28), so a second camera on the same corridor prevents less than the first. Use a minimum spacing, or a benefit that shrinks near existing cameras, so the ranking doesn't stack cameras on one arterial.
- **Same features in every step.** Site-type features must be defined the same way in the risk model, the SPF and the effect grouping. For example, if foot traffic defines a type, all three use the same NYC Walks volume.
- **Eligibility comes last.** Sites aren't filtered by school zone. The output is a ranked list that DOT screens for zone eligibility and the 750-zone cap.

## Still open
- The 3–4 site types for the effect step.
- Which layer holds the benefit calculation and the placement comparison (D, E).
- Injury-crash and pedestrian-injury robustness checks: `NegativeBinomialModel` doesn't yet converge on those outcomes with the full road feature set.

## References
- Hauer, E. (1997). *Observational Before–After Studies in Road Safety.* Pergamon. The EB before–after method.
- Nguyen (2026). "Learning to allocate automated speed enforcement." *Case Studies on Transport Policy.* Source of the risk × effect placement comparison and stress test.
- Sevtsuk, A. et al. "Spatial Distribution of Foot-traffic in New York City and Applications for Urban Planning." *Nature Cities.* doi:10.1038/s44284-025-00383-y. The NYC Walks pedestrian volumes.
