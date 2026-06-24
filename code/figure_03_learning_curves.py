import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import analysis_results as A, figure_common as F

LC_IC  = str(A.PACKAGE_ROOT / "data" / "learning_curves" / "full_incountry_learning_curve.csv")
LC_OOC = str(A.PACKAGE_ROOT / "data" / "learning_curves" / "full_outofcountry_learning_curve.csv")
LC_WA  = str(A.PACKAGE_ROOT / "data" / "learning_curves" / "west_africa_singlepatch_learning_curve.csv")
LC_MIL = str(A.PACKAGE_ROOT / "data" / "learning_curves" / "west_africa_mil_learning_curve.csv")

def figure_learning_curves():
    if not all(os.path.exists(p) for p in [LC_IC, LC_OOC, LC_WA, LC_MIL]):
        print("[SKIP] learning-curve CSVs not found"); return None
    d = A.load_predictions(); s = A.r2_summary(d)
    lc_ic, lc_ooc, lc_wa, lc_mil = (pd.read_csv(p) for p in [LC_IC, LC_OOC, LC_WA, LC_MIL])
    def panel(ax, curves, pooled, title, xt):
        for x, y, c, lab in curves:
            xs, ys = F.smooth(x, np.array(y, float), clip_lo=0.0); ax.plot(xs, ys, color=c, lw=2.0, label=lab)
        if np.isfinite(pooled): ax.axhline(pooled, color=F.C_DARK, lw=1.2, ls="--", alpha=0.7)
        ax.set_title(title, fontsize=13, fontweight="bold"); ax.set_xlabel("Training Fraction"); ax.set_ylabel("R\u00b2")
        ax.set_xticks(xt); ax.set_xticklabels([F.PCT.get(v, str(v)) for v in xt]); ax.set_ylim(0, 0.85); ax.legend(fontsize=9); ax.grid(True, alpha=0.3, ls="--")
    fig, ax = plt.subplots(2, 2, figsize=(14, 9)); fig.suptitle("Learning Curves - All Four Models", fontsize=15, fontweight="bold")
    ft = lc_ic["frac_train"].values
    panel(ax[0,0], [(ft, lc_ic["ms_r2_mean"], F.C_MS, "MS"), (ft, lc_ic["nl_r2_mean"], F.C_NL, "NTL"), (ft, lc_ic["fused_r2_mean"], F.C_FUSED, "Fused")], s["pooled"]["IC"]["fused"], "Full Dataset - In-Country", ft)
    ft2 = lc_ooc["frac_train"].values
    panel(ax[0,1], [(ft2, lc_ooc["ms_r2_pooled"], F.C_MS, "MS"), (ft2, lc_ooc["nl_r2_pooled"], F.C_NL, "NTL"), (ft2, lc_ooc["fused_r2_pooled"], F.C_FUSED, "Fused")], s["pooled"]["OOC"]["fused"], "Full Dataset - Out-of-Country", ft2)
    w = lc_wa[lc_wa["scheme"]=="incountry"].sort_values("frac_train"); ft3 = sorted(w["frac_train"].unique())
    panel(ax[1,0], [(w[w["mode"]==m]["frac_train"], w[w["mode"]==m]["pooled_test_r2"], c, l) for m,c,l in [("ms",F.C_MS,"MS"),("nl",F.C_NL,"NTL"),("fused",F.C_FUSED,"Fused")]], s["pooled"]["WA"]["fused"], "West Africa - Single-Patch", ft3)
    mi = lc_mil[lc_mil["scheme"]=="incountry"].copy(); mi["mode"] = mi["mode"].str.lower(); ft4 = sorted(mi["frac_train"].unique())
    panel(ax[1,1], [(mi[mi["mode"]==m].groupby("frac_train")["test_r2_mean"].mean().index, mi[mi["mode"]==m].groupby("frac_train")["test_r2_mean"].mean().values, c, l) for m,c,l in [("ms",F.C_MS,"MS"),("nl",F.C_NL,"NTL"),("fused",F.C_FUSED,"Fused")]], s["pooled"]["MIL"]["fused"], "West Africa - MIL 3\u00d73", ft4)
    plt.tight_layout(rect=[0,0,1,0.96]); return F.save(fig, "figure_learning_curves.png")

if __name__ == "__main__":
    figure_learning_curves()
