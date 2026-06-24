import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib as mpl
import analysis_results as A, figure_common as F

def figure_africa_maps(s):
    world = F.basemap(); ncol = next((c for c in ["ISO_A3","ADM0_A3","iso_a3"] if c in world.columns), None)
    world["iso3"] = world[ncol].str.upper(); africa = world.cx[-25:60, -38:38]
    maps = [("In-Country (Full Dataset)", s["country"]["IC"]["fused"]), ("Out-of-Country (Full Dataset)", s["country"]["OOC"]["fused"]),
            ("Single-Patch (West Africa)", s["country"]["WA"]["fused"])]
    if "MIL" in s["country"]: maps.append(("MIL 3\u00d73 (West Africa)", s["country"]["MIL"]["fused"]))
    fig, axes = plt.subplots(2, 2, figsize=(16, 16)); fig.suptitle("Fused R\u00b2 by Country - Africa Map", fontsize=15, fontweight="bold", y=1.01)
    _ISO2CC = {v: k for k, v in A.CC2ISO3.items()}
    for ax, (title, dct) in zip(axes.flatten(), maps):
        ax.set_facecolor("#D6EAF8")
        rdf = pd.DataFrame([{"iso3": A.CC2ISO3[c], "r2": v} for c, v in dct.items() if c in A.CC2ISO3])
        merged = africa.merge(rdf, on="iso3", how="left"); hd = merged[merged["r2"].notna()]
        africa.plot(ax=ax, color="#CCCCCC", edgecolor="#888", linewidth=0.4)
        if not hd.empty:
            hd.plot(ax=ax, column="r2", cmap="RdYlGn", vmin=0.0, vmax=0.85, edgecolor="#555", linewidth=0.5)
            for _, row in hd.iterrows():
                cc = _ISO2CC.get(row["iso3"], "")
                if cc:
                    rp = row["geometry"].representative_point()
                    ax.annotate(cc, (rp.x, rp.y), ha="center", va="center", fontsize=9, fontweight="bold", color="#1a1a1a")
        ax.set_xlim(-25,55); ax.set_ylim(-38,38); ax.set_title(title, fontsize=13, fontweight="bold"); ax.set_axis_off()
    sm = mpl.cm.ScalarMappable(cmap="RdYlGn", norm=mpl.colors.Normalize(0.0, 0.85)); sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, orientation="horizontal", fraction=0.045, pad=0.04, shrink=0.5)
    cb.set_label("Fused R\u00b2")
    return F.save(fig, "figure_africa_maps.png")

if __name__ == "__main__":
    d = A.load_predictions(); figure_africa_maps(A.r2_summary(d))
