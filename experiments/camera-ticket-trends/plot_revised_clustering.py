"""Review figures from #58 outputs, using the shared team chart style."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import NullLocator

from viz.style import (
    CATEGORICAL,
    COMPARISON,
    FULL_WIDTH,
    INK_2,
    MODEL,
    MUTED,
    use_style,
)


def plot_results(directory, destination):
    use_style()
    destination.mkdir(parents=True, exist_ok=True)
    outcomes = pd.read_csv(directory / "camera_outcomes.csv").set_index("camera_id")
    panel = pd.read_parquet(directory / "corrected_first_year.parquet")
    indexed = panel.div(outcomes.early_mean, axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH * 1.5, 3.7), sharey=True)
    for axis, half in zip(axes, ["A", "B"]):
        rows = outcomes.loc[outcomes.split.eq(half) & outcomes.cluster.notna()]
        for color, (cluster, group) in zip(CATEGORICAL, rows.groupby("cluster")):
            curve = indexed.loc[group.index].median().rolling(4, min_periods=1).mean()
            axis.plot(range(1, len(curve) + 1), curve, color=color)
            axis.annotate(f"C{int(cluster)} (n={len(group):,})", (52, curve.iloc[-1]),
                          xytext=(5, 0), textcoords="offset points", color=INK_2, va="center")
        axis.set(title=f"{half}: {'discovery' if half == 'A' else 'frozen-model holdout'}", xlabel="Weeks since activation", xlim=(1, 70))
        axis.set_yscale("log")
        axis.set_yticks([.5, 1, 1.5], ["0.5×", "1×", "1.5×"])
        axis.set_ylim(.4, 1.5)
        axis.yaxis.set_minor_locator(NullLocator())
        axis.set_xticks([1, 13, 26, 39, 52])
        axis.axhline(1, color=MUTED, linestyle="--", linewidth=1)
    axes[0].set_ylabel("Median corrected rate / own weeks 1–13 mean (log scale)")
    fig.suptitle("The two A-defined shapes also separate in B")
    fig.text(.01, .01, "Source: frozen NYC DOF camera tickets + team #23 registry/schedule. Four-week display smoothing; descriptive, not causal.", fontsize=8)
    fig.tight_layout(rect=(0, .08, 1, .93))
    fig.savefig(destination / "corrected_shapes.png")
    plt.close(fig)

    results = pd.read_csv(directory / "feature_validation.csv")
    labels = {"lanes": "Lanes: above A median", "width_ft": "Width: above A median", "ped_volume": "Foot traffic: above A median"}
    names = [labels.get(row.feature, row.group + " vs two-way × two-way") for row in results.itertuples()]
    fig, axis = plt.subplots(figsize=(FULL_WIDTH * 1.5, 3.8))
    for half, offset, color in [("A", -.12, MODEL), ("B", .12, COMPARISON)]:
        values = results[f"{half}_difference_pp"]
        errors = np.vstack([values - results[f"{half}_ci_low"], results[f"{half}_ci_high"] - values])
        axis.errorbar(values, np.arange(len(results)) + offset, xerr=errors, fmt="o", color=color, label=half, capsize=3)
    axis.set_yticks(range(len(names)), names)
    axis.invert_yaxis()
    axis.axvline(0, color=MUTED, linestyle="--", linewidth=1)
    axis.set(title="No decline contrast passes the B holdout check", xlabel="Difference in mean 52-week decline (percentage points; approximate 95% CI)")
    axis.legend(title="Camera half")
    fig.text(.01, .01, "Source: NYC DOF, DCP LION, MIT NYC Walks. Same borough × activation-year strata and A weights in both halves.", fontsize=8)
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(destination / "feature_holdout.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "experiments/camera-ticket-trends/outputs/issue58")
    parser.add_argument("--destination", type=Path, default=Path(__file__).parent / "issue58-review")
    args = parser.parse_args()
    plot_results(args.directory, args.destination)
