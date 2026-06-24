import pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import analysis_results as A, figure_common as F

def figure_fold_maps():
    clu = pd.read_csv(A.CONFIG.CLUSTERS_CSV)
    clu["key"] = clu["country"].astype(str)+"|"+clu["year"].astype(int).astype(str)+"|"+clu["cluster"].astype(int).astype(str)
    coords = clu[["key","lat","lon"]]
    ic  = pd.read_csv(A.CONFIG.PREDS_IC)[["key","fold"]].merge(coords, on="key")
    ooc = pd.read_csv(A.CONFIG.PREDS_OOC)[["key","fold"]].merge(coords, on="key")
    world = F.basemap()
    def draw_one(df, title, outname):
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.set_facecolor("#a9c5de"); world.plot(ax=ax, color="#f4efd5", edgecolor="#8f8f82", linewidth=0.4)
        for f in "ABCDE":
            s = df[df["fold"] == f]; ax.scatter(s["lon"], s["lat"], s=2.2, c=F.FOLD_COLOURS[f], linewidths=0, alpha=0.85)
        ax.set_xlim(-23,58); ax.set_ylim(-37,41); ax.set_aspect("equal")
        ax.set_title(title, fontsize=13); ax.set_xlabel("Longitude", fontsize=12); ax.set_ylabel("Latitude", fontsize=12); ax.tick_params(labelsize=11)
        h = [Line2D([0],[0], marker="o", color="w", markerfacecolor=F.FOLD_COLOURS[f], markersize=6, label=f) for f in "ABCDE"]
        ax.legend(handles=h, loc="center left", bbox_to_anchor=(1.01,0.5), frameon=False, fontsize=11, labelspacing=0.6)
        plt.tight_layout(); return F.save(fig, outname)
    draw_one(ic,  "DHS in-country TEST folds (A\u2013E)", "figure_fold_maps_ic.png")
    return draw_one(ooc, "DHS Out-of-Country TEST folds (A\u2013E)", "figure_fold_maps_ooc.png")

if __name__ == "__main__":
    figure_fold_maps()
