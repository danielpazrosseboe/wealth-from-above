import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import analysis_08_rwi_benchmark as B
import figure_common as F

GREEN = "#3a9b4e"
GREEN_FILL = "#5cb85c"
RWI_C = "#d98b30"
DASH = "#2f7d3a"
ORDER = ["ic", "ooc", "sp", "mil"]
XLAB = ["Full\nin-country", "Full\nout-of-country", "West Africa\nsingle-patch", "West Africa\nMIL 3x3"]


def _strip(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def figure_bars(agg):
    x = np.arange(4); w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    for ax, (key, title) in zip(axes, [("r2", "Standardized R\u00b2 (squared Pearson r)"),
                                        ("rho", "Spearman \u03c1 (rank correlation)")]):
        mv = [agg[c]["m_" + key] for c in ORDER]
        rv = [agg[c]["r_" + key] for c in ORDER]
        b1 = ax.bar(x - w / 2, mv, w, label="Fused model", color=GREEN, edgecolor="white", lw=0.5)
        b2 = ax.bar(x + w / 2, rv, w, label="Relative Wealth Index", color=RWI_C, edgecolor="white", lw=0.5)
        for b in list(b1) + list(b2):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.013, f"{b.get_height():.2f}",
                    ha="center", va="bottom", fontsize=11)
        _strip(ax); ax.set_axisbelow(True); ax.yaxis.grid(True, color="#e6e9ec", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(XLAB, fontsize=11); ax.set_ylim(0, 0.9)
        ax.set_title(title, fontweight="bold")
    axes[0].legend(frameon=True, loc="upper right", fontsize=11, edgecolor="#cccccc")
    fig.suptitle("Benchmarking Against the Relative Wealth Index - By Configuration",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    return F.save(fig, "figure_rwi_bars.png")


def figure_per_country(per):
    t = per["ic"].sort_values("m_r2").reset_index(drop=True)
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(8.4, 8.6))
    for i in y:
        ax.plot([t["r_r2"][i], t["m_r2"][i]], [i, i], color="#c5ccd2", lw=2, zorder=1)
    ax.scatter(t["r_r2"], y, s=58, color=RWI_C, zorder=2, label="Relative Wealth Index")
    ax.scatter(t["m_r2"], y, s=58, color=GREEN, zorder=3, label="Fused model")
    ax.set_yticks(y); ax.set_yticklabels(t["name"], fontsize=11); ax.set_ylim(-1, len(t))
    ax.set_xlim(0, 1); ax.set_xlabel("Within-country variance explained, R\u00b2")
    _strip(ax); ax.set_axisbelow(True); ax.xaxis.grid(True, color="#e6e9ec", lw=0.8)
    ax.legend(frameon=True, loc="lower right", fontsize=11, edgecolor="#cccccc")
    ax.set_title("Per-Country Comparison - Model vs Relative Wealth Index (In-Country)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    return F.save(fig, "figure_rwi_per_country.png")


def figure_scatter(raw, cc="NG"):
    ic = raw["ic"]
    g = ic[(ic["country"] == cc)].dropna(subset=["rwi"]).copy()
    rwi_z = (g["rwi"] - g["rwi"].mean()) / g["rwi"].std()
    name = B.A.CC2NAME.get(cc, cc)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), sharex=True)

    def panel(ax, xv, yv, title, ylab):
        ax.set_facecolor("#fbfcfd")
        ax.scatter(xv, yv, s=16, color=GREEN_FILL, alpha=0.55, edgecolor="none")
        ax.plot([-1.7, 3], [-1.7, 3], "--", color="black", lw=1.3)
        r2 = pearsonr(xv, yv)[0] ** 2
        ax.text(0.04, 0.94, f"R\u00b2 = {r2:.4f}\nn = {len(xv):,}", transform=ax.transAxes,
                va="top", fontsize=11, fontweight="bold", color=DASH,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#cccccc"))
        _strip(ax); ax.set_title(title, fontweight="bold")
        ax.set_xlabel("True Wealth Index"); ax.set_ylabel(ylab)
        ax.set_xlim(-1.7, 3); ax.set_ylim(-1.7, 3)

    panel(axes[0], g["y_true"].values, g["yhat_fused"].values, "Fused model", "Predicted (Fused)")
    panel(axes[1], g["y_true"].values, rwi_z.values, "Relative Wealth Index", "RWI (standardized)")
    fig.suptitle(f"Predicted vs True Wealth Index - {name} (Model vs RWI)",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    return F.save(fig, "figure_rwi_scatter.png")


if __name__ == "__main__":
    per, agg, raw = B.run()
    figure_bars(agg)
    figure_per_country(per)
    figure_scatter(raw, "NG")
