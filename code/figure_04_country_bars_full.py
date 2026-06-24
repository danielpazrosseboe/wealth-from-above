import pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import analysis_results as A, figure_common as F

def _cr(d_cc): return pd.DataFrame([{"cc": cc, "name": A.CC2NAME.get(cc, cc), "fused": v} for cc, v in d_cc.items()])

def figure_country_bars_full(s):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle("Country-Level Fused R\u00b2 - Full Dataset (22 Countries)", fontsize=15, fontweight="bold", y=1.01)
    F.hbar_panel(ax[0], _cr(s["country"]["IC"]["fused"]),  "fused", s["pooled"]["IC"]["fused"],  "In-Country")
    F.hbar_panel(ax[1], _cr(s["country"]["OOC"]["fused"]), "fused", s["pooled"]["OOC"]["fused"], "Out-of-Country")
    plt.tight_layout(); return F.save(fig, "figure_country_bars_full.png")

if __name__ == "__main__":
    d = A.load_predictions(); figure_country_bars_full(A.r2_summary(d))
