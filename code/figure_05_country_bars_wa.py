import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import analysis_results as A, figure_common as F

def figure_country_bars_wa(s):
    if "MIL" not in s["pooled"]: print("[SKIP] MIL predictions absent"); return None
    order = sorted(A.WA, key=lambda c: s["country"]["MIL"]["fused"].get(c, -99))
    cr_wa  = pd.DataFrame([{"cc":c,"name":A.CC2NAME.get(c,c),"fused":s["country"]["WA"]["fused"].get(c, np.nan)} for c in order])
    cr_mil = pd.DataFrame([{"cc":c,"name":A.CC2NAME.get(c,c),"fused":s["country"]["MIL"]["fused"].get(c, np.nan)} for c in order])
    fig, ax = plt.subplots(1, 2, figsize=(17, 7))
    fig.suptitle("Country-Level Fused R\u00b2 - West Africa: Single-Patch vs MIL 3\u00d73", fontsize=15, fontweight="bold", y=1.01)
    F.hbar_panel(ax[0], cr_wa,  "fused", s["pooled"]["WA"]["fused"],  "Single-Patch (In-Country)", fs=9)
    F.hbar_panel(ax[1], cr_mil, "fused", s["pooled"]["MIL"]["fused"], "MIL 3\u00d73 (In-Country)", fs=9)
    plt.tight_layout(); return F.save(fig, "figure_country_bars_wa.png")

if __name__ == "__main__":
    d = A.load_predictions(); figure_country_bars_wa(A.r2_summary(d))
