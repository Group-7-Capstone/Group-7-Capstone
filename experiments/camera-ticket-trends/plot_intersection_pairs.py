"""Export research figures and editable Data chart inputs from reviewed road-pair tables."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
TITLES = {"direction":"Traffic direction", "width":"Paved width", "lanes":"Travel lanes"}
PALETTE = ["#165AA7","#AE5013","#008479","#7C4998","#A8255D","#546B23",
           "#6D5C43","#2384AF","#52698A","#5A535E"]


def figures(directory):
    weekly = pd.read_csv(directory/"pair_weekly_trends.csv")
    weekly = weekly.loc[weekly.road_pair.ne("Unknown / ambiguous")].copy()
    summary = pd.read_csv(directory/"pair_summary.csv")
    for dim, title in TITLES.items():
        data = weekly.loc[weekly.dimension.eq(dim)]
        colors = dict(zip(sorted(data.road_pair.unique()), PALETTE))
        fig, axes = plt.subplots(1, 2, figsize=(14, 7.1))
        fig.subplots_adjust(left=.07,right=.98,top=.78,bottom=.37,wspace=.22)
        fig.suptitle(title+" at intersecting streets",x=.07,y=.965,ha="left",fontsize=23,fontweight="bold",color="#132E45")
        fig.text(.07,.881,"Weekly tickets at each location, indexed to its first 13 weeks (100). Lines show trailing four-week means.",fontsize=11)
        for ax, kind in zip(axes, ["Speed","Red Light"]):
            part = data.loc[data.camera_type.eq(kind)]
            sizes = summary.loc[summary.camera_type.eq(kind) & summary.dimension.eq(dim)]
            total = int(sizes.n.sum())
            known = int(sizes.loc[sizes.road_pair.ne("Unknown / ambiguous"),"index_n"].sum())
            ax.set_title(f"{kind} cameras   ·   {known:,} classified / {total:,} eligible",fontsize=12,loc="left")
            ax.axhline(100,color="#818990",linestyle=":",linewidth=1)
            rows = []
            for group, group_rows in part.groupby("road_pair",sort=True):
                group_rows = group_rows.sort_values("week")
                n = int(group_rows.index_n.iloc[0])
                label = f"{group}  (n={n:,})"
                ax.plot(group_rows.week,group_rows.index_mean_4w,color=colors[group],label=label,
                        linewidth=2.2 if n>=10 else 1.4,linestyle="-" if n>=10 else "--")
                for r in group_rows.itertuples():
                    rows.append({"week":int(r.week),"group":label,"index":None if pd.isna(r.index_mean_4w) else float(r.index_mean_4w)})
            ax.set_xlim(1,52)
            ax.set_xticks([1,13,26,39,52])
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Week since inferred activation")
            ax.set_ylabel("Mean ticket index")
            ax.grid(axis="y",color="#E0E6E9",linewidth=.7)
            ax.spines[["top","right"]].set_visible(False)
            ax.legend(loc="upper left",bbox_to_anchor=(-.02,-.23),ncol=2,fontsize=8.5,frameon=False)
            chart = {"id":f"road-pairs-{dim}-{kind.lower().replace(' ','-')}",
                     "title":f"{kind}: {title} at intersecting streets","height":425,
                     "description":f"{known} classified of {total} eligible groups. Four-week means; own first 13-week mean = 100.",
                     "rows":rows,"spec":{"type":"line","x":"week","y":"index","series":"group",
                     "xLabel":"Week since inferred activation","yLabel":"Mean ticket index","xDomain":[1,52]}}
            (directory/f"{dim}_{kind.lower().replace(' ','_')}.chart.json").write_text(
                json.dumps(chart,allow_nan=False,ensure_ascii=False),encoding="utf-8")
        fig.text(.07,.084,"n = address–type groups with 52 complete weeks and a resolved pair. Dashed curves have fewer than 10 groups. Panels use separate y scales.",fontsize=9,color="#425563")
        fig.text(.07,.049,"Source: frozen NYC ticket histories and LION named intersections. Both street orders share one category. Width cutoff: 15 m.",fontsize=9,color="#425563")
        fig.savefig(directory/f"{dim}_pair_trends.png",dpi=180,facecolor="white")
        plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=HERE/"outputs"/"intersection_pairs")
    figures(p.parse_args().output)
