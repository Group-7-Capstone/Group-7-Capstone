"""Static notebook figures from the same reviewed tables as the dashboard."""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

COLORS = ["#2864a5", "#c46a25", "#60804c", "#98658d", "#958128", "#5d8f91"]


def finish(fig, destination, name):
    fig.tight_layout(rect=(0, .04, 1, .96))
    fig.text(.02, .01, "Frozen snapshot · Complete calendar weeks · Descriptive ticket counts, not rates or causal effects", fontsize=9, color="#53616d")
    path = destination / "figures"
    path.mkdir(exist_ok=True)
    fig.savefig(path / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def axes_pair(title, height=4.5):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, height))
    fig.suptitle(title, fontsize=15, weight="bold")
    for ax, kind in zip(axes, ["Red Light", "Speed"]):
        ax.set_title(kind, loc="left")
        ax.grid(axis="y", alpha=.16)
    return fig, axes


def plot_eligibility(tables, out):
    fig, axes = axes_pair("Eligibility from inferred activation")
    for ax, kind in zip(axes, ["Red Light", "Speed"]):
        group = tables["coverage"].query("camera_type == @kind")
        fields = ["groups", "eligible_26w", "eligible_52w", "eligible_104w", "eligible_administrative_52w"]
        counts = group[fields].sum().to_numpy()
        ax.barh(["All original groups", "26 complete weeks", "52 complete weeks", "104 complete weeks", "52 weeks: fixed cutoff"], counts, color=COLORS[0])
        ax.invert_yaxis()
        ax.set_xlim(0, max(counts)*1.18)
        for i, count in enumerate(counts): ax.text(count+max(counts)*.02, i, f"{count:,}", va="center")
        ax.set_xlabel("Enforcement groups")
    finish(fig, out, "eligibility")


def plot_shared(tables, out):
    frame = tables["shared_trends"].query("scenario == 'balanced_52'")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Shared ticket trajectories: fixed 52-week cohorts", fontsize=15, weight="bold")
    for column, kind in enumerate(["Red Light", "Speed"]):
        subset = frame.query("camera_type == @kind")
        for row, metric in enumerate(["raw", "index"]):
            ax = axes[row, column]
            ax.fill_between(subset.week, subset[f"{metric}_q25"], subset[f"{metric}_q75"], color="#bcc4cc", alpha=.35, label="Camera interquartile range")
            ax.fill_between(subset.week, subset[f"{metric}_ci_low"], subset[f"{metric}_ci_high"], color=COLORS[0], alpha=.22, label="Pointwise 95% bootstrap CI")
            ax.plot(subset.week, subset[f"{metric}_mean"], color=COLORS[0], label="Mean", lw=1.8)
            ax.plot(subset.week, subset[f"{metric}_median"], color=COLORS[1], label="Median", lw=1.5)
            ax.plot(subset.week, subset[f"{metric}_mean_4w"], color="#303b46", ls="--", label="Four-week mean")
            n = int(subset.n.iloc[0] if metric == "raw" else subset.index_n.iloc[0])
            ax.set_title(f"{kind} · n={n:,}", loc="left")
            ax.set_xlabel("Weeks from inferred activation (first full week = 1)")
            ax.set_ylabel("Tickets / camera / week" if metric == "raw" else "Own first 13-week mean = 100")
            ax.set_ylim(bottom=0)
            ax.grid(axis="y", alpha=.16)
            if row == 0 and column == 0: ax.legend(fontsize=8)
    finish(fig, out, "shared_trends")


def plot_sensitivities(tables, out):
    from validation.camera_shared_trends import SCENARIOS
    fig, axes = axes_pair("Normalized-trend sensitivity checks", 5)
    keys = ["balanced_52", "high_confidence", "administrative", "skip_first_four", "baseline_at_least_five"]
    for ax, kind in zip(axes, ["Red Light", "Speed"]):
        for color, scenario in zip(COLORS, keys):
            s = tables["shared_trends"].query("camera_type == @kind and scenario == @scenario")
            if len(s): ax.plot(s.week, s.index_mean, color=color, label=f"{SCENARIOS[scenario]} (n={int(s.index_n.iloc[0]):,})")
        ax.axhline(100, color="#77818b", ls=":")
        ax.set_xlabel("Relative week")
        ax.set_ylabel("Mean index (own 13-week baseline = 100)")
        ax.legend(fontsize=7.5)
    finish(fig, out, "sensitivities")


def plot_calendar(tables, out):
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex="col")
    fig.suptitle("Calendar-time trends and changing contributors", fontsize=15, weight="bold")
    for i, kind in enumerate(["Red Light", "Speed"]):
        s = tables["calendar_trends"].query("camera_type == @kind")
        dates = pd.to_datetime(s.week_start)
        axes[0, i].plot(dates, s.raw_mean, color=COLORS[0], label="Mean")
        axes[0, i].plot(dates, s.raw_median, color=COLORS[1], label="Median")
        axes[0, i].set_title(kind, loc="left")
        axes[0, i].set_ylabel("Tickets / camera / week")
        axes[0, i].legend()
        axes[1, i].plot(dates, s.n, color=COLORS[0])
        axes[1, i].set_ylabel("Contributing groups")
        axes[1, i].xaxis.set_major_locator(mdates.YearLocator(2))
        axes[1, i].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        for ax in axes[:, i]: ax.set_ylim(bottom=0); ax.grid(axis="y", alpha=.16)
    finish(fig, out, "calendar_trends")


def plot_cohorts(tables, out):
    fig, axes = axes_pair("52-week trajectories by inferred activation year", 5)
    for ax, kind in zip(axes, ["Red Light", "Speed"]):
        frame = tables["shared_trends"]
        frame = frame[frame.camera_type.eq(kind) & frame.scenario.str.startswith("cohort_")]
        for i, (scenario, s) in enumerate(frame.groupby("scenario")):
            ax.plot(s.week, s.index_mean_4w, color=COLORS[i % len(COLORS)], ls="--" if i >= 6 else "-", label=f"{scenario[7:]} (n={int(s.index_n.iloc[0]):,})")
        ax.set_ylabel("Four-week mean index")
        ax.set_xlabel("Relative week")
        ax.legend(fontsize=8, ncol=2)
    finish(fig, out, "activation_cohorts")


def plot_diagnostics(tables, out):
    fig, axes = axes_pair("Candidate cluster diagnostics")
    for ax, kind in zip(axes, ["Red Light", "Speed"]):
        s = tables["cluster_diagnostics"].query("camera_type == @kind")
        ax.plot(s.k, s.silhouette, "o-", color=COLORS[0], label="Silhouette")
        ax.plot(s.k, s.stability_ari, "s-", color=COLORS[1], label="Mean stability ARI")
        chosen = s[s.selected]
        if len(chosen): ax.axvline(chosen.k.iloc[0], ls=":", color="#596574", label="Selected K")
        ax.set_xticks(range(2, 7)); ax.set_ylim(0, 1.05)
        ax.set_xlabel("Number of clusters"); ax.legend()
    finish(fig, out, "cluster_diagnostics")


def plot_clusters(tables, out):
    fig, axes = axes_pair("Shape-cluster centroids: first 52 complete weeks", 5)
    for ax, kind in zip(axes, ["Red Light", "Speed"]):
        frame = tables["cluster_curves"]
        subset = frame.query("camera_type == @kind") if not frame.empty else frame
        if subset.empty:
            ax.set_axis_off()
            ax.text(.05, .62, "No cluster solution accepted", fontsize=15, weight="bold", transform=ax.transAxes)
            ax.text(.05, .40, "Candidate partitions did not pass all\nseparation, stability and size thresholds.\nSee model-selection diagnostics.", fontsize=12, linespacing=1.7, transform=ax.transAxes)
            continue
        for i, (label, s) in enumerate(subset.groupby("cluster_label")):
            ax.plot(s.week, s.shape_mean, color=COLORS[i], label=f"{label} (n={int(s.n.iloc[0]):,})")
        ax.axhline(0, color="#89929c", lw=.7)
        ax.set_xlabel("Relative week")
        ax.set_ylabel("Standardized log four-week mean")
        ax.legend(fontsize=9)
    finish(fig, out, "cluster_centroids")
