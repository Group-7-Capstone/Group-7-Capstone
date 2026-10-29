"""Publication figures for the frozen per-camera weekly summaries."""
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter, MaxNLocator
import pandas as pd

plt.rcParams.update({"figure.dpi":110, "savefig.dpi":300, "font.size":10,
                     "axes.spines.top":False, "axes.spines.right":False,
                     "axes.titleweight":"bold", "axes.titlelocation":"left"})
BLUE, ORANGE = "#2962a3", "#bf5b12"


def select_examples(summary):
    examples = []
    for kind in ["Red Light", "Speed"]:
        rows = summary[summary.camera_type.eq(kind) & summary.analysis_ticket_count.gt(0)]
        rows = rows.sort_values(["analysis_ticket_count", "camera_id"])
        for rank, position in [("Low",0),("Median",len(rows)//2),("High",len(rows)-1)]:
            row = rows.iloc[position].copy()
            row["example_volume"] = rank
            examples.append(row)
    return pd.DataFrame(examples).reset_index(drop=True)


def plot_group(row, weekly, destination):
    destination = Path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    fig, ax = plt.subplots(figsize=(12,5.8))
    fig.subplots_adjust(top=.73,bottom=.24,left=.09,right=.96)
    activation = pd.Timestamp(row.inferred_activation_date)
    last = pd.to_datetime(row.last_actual_ticket_date)
    if len(weekly):
        ax.plot(weekly.week_start,weekly.ticket_count,color=BLUE,lw=.9,alpha=.75,
                marker="o" if len(weekly)<26 else None,markersize=5,label="Weekly unique tickets")
        if weekly.moving_average_4w.notna().any():
            ax.plot(weekly.week_start,weekly.moving_average_4w,color=ORANGE,lw=1.8,label="4-week mean (complete weeks)")
        for week in weekly.loc[weekly.is_partial_week,"week_start"]:
            ax.axvspan(week,week+pd.Timedelta(days=6),color="#efbd63",alpha=.4)
        for week in weekly.loc[~weekly.source_complete,"week_start"]:
            ax.axvspan(week,week+pd.Timedelta(days=6),color="#a5abb1",alpha=.25)
        ax.axvline(activation,color="#43505f",lw=1,ls="--",label="Inferred activation")
        ax.axvline(last,color="#43505f",lw=1,ls=":",label="Last actual ticket")
        ax.set_xlim(min(activation,weekly.week_start.min())-pd.Timedelta(days=12),last+pd.Timedelta(days=12))
    else:
        ax.text(.5,.5,"No matched post-activation tickets",ha="center",transform=ax.transAxes)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Unique tickets per calendar week")
    ax.set_xlabel("Calendar week (Monday start)")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6,integer=True))
    locator=mdates.AutoDateLocator(minticks=4,maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    ax.grid(axis="y",alpha=.18)
    address="\n".join(textwrap.wrap(str(row.search_query),width=95))
    fig.text(.09,.95,f"{row.camera_type} · {row.get('example_volume','Selected')} volume · {row.detection_confidence}",fontsize=11,color="#526170")
    fig.text(.09,.90,address,fontsize=13,weight="bold",va="top")
    span=f"{str(activation.date())} to {str(last.date()) if pd.notna(last) else 'no last ticket'}"
    fig.text(.09,.79,f"{row.camera_id}   |   {int(row.analysis_ticket_count):,} tickets   |   {span}   |   {int(row.zero_ticket_weeks)} zero-ticket weeks",fontsize=9)
    handles,labels=ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles,labels,loc="lower left",bbox_to_anchor=(.08,.10),ncol=2,frameon=False,fontsize=9)
    fig.text(.09,.06,"Amber shading: partial boundary weeks. Gray shading: unavailable source coverage. No tickets does not prove equipment inactivity.",fontsize=8,color="#526170")
    fig.text(.09,.025,"Source: NYC Open Data FY2017–FY2027, issue dates 2016-01-01–2026-08-30. Eric's original grouping and activation dates preserved.",fontsize=8,color="#526170")
    fig.savefig(destination/f"{row.camera_id}.png",facecolor="white")
    plt.show()
    plt.close(fig)


def plot_overview(summary,weekly,destination):
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
    joined=weekly.merge(summary[["camera_id","camera_type"]],on="camera_id",validate="many_to_one")
    totals=joined.groupby(["camera_type","week_start"]).agg(tickets=("ticket_count",lambda x:x.sum(min_count=1)),groups=("camera_id","nunique")).reset_index()
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True,gridspec_kw={"height_ratios":[2,1]})
    for kind,color in [("Speed",BLUE),("Red Light",ORANGE)]:
        frame=totals[totals.camera_type.eq(kind)]
        axes[0].plot(frame.week_start,frame.tickets,color=color,lw=1.3,label=kind)
        axes[1].plot(frame.week_start,frame.groups,color=color,lw=1.3,label=kind)
    axes[0].set_title("Ticket counts across the frozen enforcement groups")
    axes[0].set_ylabel("Unique tickets / week")
    axes[1].set_ylabel("Groups with an\nobservation window")
    axes[1].set_xlabel("Calendar week (Monday start)")
    for ax in axes:
        ax.set_ylim(bottom=0); ax.grid(axis="y",alpha=.18)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes[0].legend(frameon=False)
    locator=mdates.AutoDateLocator(minticks=6,maxticks=11)
    axes[1].xaxis.set_major_locator(locator); axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.subplots_adjust(left=.1,right=.96,top=.92,bottom=.16,hspace=.14)
    fig.text(.1,.065,"Counts include partial boundary weeks. The number of groups changes over time; this is not a balanced-panel trend or a violation rate.",fontsize=9,color="#526170")
    fig.text(.1,.03,"Source: fresh NYC Open Data FY2017–FY2027 extracts, capped at August 30, 2026; unchanged Eric registry.",fontsize=9,color="#526170")
    fig.savefig(destination/"overall_ticket_history.png",facecolor="white")
    plt.show(); plt.close(fig)
