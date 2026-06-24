import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import analysis_results as A, figure_common as F

def figure_district_bars():
    d = A.load_predictions()
    try:
        res = A.district_by_country_wa(d)
    except Exception as e:
        print(f"[SKIP] district bars - inputs missing: {e}"); return None
    if "MIL" not in res:
        print("[SKIP] MIL absent"); return None

    def frame(per):
        return pd.DataFrame([{"name": A.CC2NAME.get(c, c), "r2": v}
                             for c, v in per.items() if np.isfinite(v)])

    sp, mil = frame(res["WA"]["per"]), frame(res["MIL"]["per"])
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle("District-Level Weighted R\u00b2 by Country - West Africa",
                 fontsize=15, fontweight="bold", y=1.01)
    F.hbar_panel(axes[0], sp,  "r2", res["WA"]["pooled"],  "Single-Patch", xlim=(-0.25, 1.0))
    F.hbar_panel(axes[1], mil, "r2", res["MIL"]["pooled"], "MIL 3\u00d73",   xlim=(-0.25, 1.0))
    plt.tight_layout()
    return F.save(fig, "figure_district_bars.png")

if __name__ == "__main__":
    figure_district_bars()
