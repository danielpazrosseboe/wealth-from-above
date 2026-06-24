import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import analysis_results as A, figure_common as F

def figure_district_panels():
    try:
        import geopandas as gpd
        meta = A._load(A.CONFIG.META_CSV)
        if meta is None: raise FileNotFoundError(A.CONFIG.META_CSV)
        d = A.load_predictions()

        if "key" in meta.columns and "key" in d["ic"].columns:
            ic_c  = d["ic"].merge(meta[["key","lat","lon"]], on="key", how="left").rename(columns={"yhat_fused":"y_pred"})
            ooc_c = d["ooc"].merge(meta[["key","lat","lon"]], on="key", how="left").rename(columns={"yhat_fused":"y_pred"})
        else:
            ic_c  = A._nearest_latlon(d["ic"].rename(columns={"yhat_fused":"y_pred"}), meta)
            ooc_c = A._nearest_latlon(d["ooc"].rename(columns={"yhat_fused":"y_pred"}), meta)
        cl, di = A.district_aggregation(d)
    except Exception as e:
        print(f"[SKIP] district panels - inputs missing: {e}"); return None
    print("  district pooled R\u00b2:", "  ".join(f"{k}={di[k]:.3f}(cl {cl[k]:.3f})" for k in cl))
    fig, ax = plt.subplots(1, 2, figsize=(20, 10)); fig.suptitle("Cluster vs district predictions (fused)", fontsize=15, fontweight="bold", y=1.01)
    for a, (dfc, lab, title) in zip(ax, [(ic_c, "IC", "In-Country"), (ooc_c, "OOC", "Out-of-Country")]):
        rng = np.random.default_rng(42); idx = rng.choice(len(dfc), min(5000, len(dfc)), replace=False)
        a.scatter(dfc["y_true"].values[idx], dfc["y_pred"].values[idx], s=4, alpha=0.2, color=F.C_FUSED, rasterized=True)
        a.plot([-1.8,3.0], [-1.8,3.0], "k--", lw=1.2, alpha=0.7); a.set_xlim(-1.8,3.0); a.set_ylim(-1.8,3.0); a.set_aspect("equal")
        a.text(0.05, 0.93, f"cluster R\u00b2={cl[lab]:.4f}\ndistrict R\u00b2={di[lab]:.4f}", transform=a.transAxes, fontsize=11, va="top", fontweight="bold", color=F.C_DARK, bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#ccc"))
        a.set_xlabel("True wealth"); a.set_ylabel("Predicted"); a.set_title(title, fontsize=12, fontweight="semibold"); a.grid(True, alpha=0.3, ls="--")
    plt.tight_layout(); return F.save(fig, "figure_district_panels.png")

if __name__ == "__main__":
    figure_district_panels()
