import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import analysis_results as A, figure_common as F

def figure_scatter(d, s):
    panels = [(d["ic"], "yhat_fused", "Full Dataset - In-Country", "IC"),
              (d["ooc"], "yhat_fused", "Full Dataset - Out-of-Country", "OOC"),
              (d["wa"], "y_pred", "West Africa - Single-Patch", "WA")]
    if "mil_f" in d: panels.append((d["mil_f"], "y_pred", "West Africa - MIL 3\u00d73", "MIL"))
    fig, axes = plt.subplots(2, 2, figsize=(13, 12)); fig.suptitle("Predicted vs True Wealth Index - All Four Models (Fused)", fontsize=15, fontweight="bold")
    for ax, (df, yc, title, key) in zip(axes.flatten(), panels):
        yt, yp = df["y_true"].values, df[yc].values
        rng = np.random.default_rng(42); idx = rng.choice(len(yt), min(6000, len(yt)), replace=False)
        ax.scatter(yt[idx], yp[idx], s=5, alpha=0.22, color=F.C_FUSED, rasterized=True)
        ax.plot([-1.8,3.0], [-1.8,3.0], "k--", lw=1.2, alpha=0.7); ax.set_xlim(-1.8,3.0); ax.set_ylim(-1.8,3.0); ax.set_aspect("equal")
        ax.text(0.05, 0.93, f"R\u00b2 = {s['pooled'][key]['fused']:.4f}\nn = {len(df):,}", transform=ax.transAxes, fontsize=11, va="top", fontweight="bold", color=F.C_DARK, bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#ccc"))
        ax.set_xlabel("True Wealth Index"); ax.set_ylabel("Predicted (Fused)"); ax.set_title(title, fontsize=13, fontweight="bold"); ax.grid(True, alpha=0.3, ls="--")
    plt.tight_layout(); return F.save(fig, "figure_scatter_all4.png")

if __name__ == "__main__":
    d = A.load_predictions(); figure_scatter(d, A.r2_summary(d))
