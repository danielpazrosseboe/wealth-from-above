import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.lines import Line2D
import figure_common as F

CAP_C, MISS_C = "#17becf", "#e377c2"

def figure_coverage(n_sim=16700, seed=4):
    rng = np.random.default_rng(seed); IMG, SP = 6.72, 1.8
    R_URB, R_RUR, R_OUT, P_URB, P_RUR = 2.0, 5.0, 10.0, 0.58, 0.41
    offs = [-SP, 0, SP]
    cap = lambda vx, vy, ctr, b: any((cx-b/2 <= vx <= cx+b/2) and (cy-b/2 <= vy <= cy+b/2) for cx in ctr for cy in ctr)
    xs, ys, capS, capM = [], [], [], []
    for _ in range(n_sim):
        rv = rng.random()
        lo, hi = (0.0, R_URB) if rv < P_URB else ((R_URB, R_RUR) if rv < P_URB+P_RUR else (R_RUR, R_OUT))
        d = np.sqrt(rng.uniform(lo**2, hi**2)); a = rng.uniform(0, 2*np.pi)
        vx, vy = d*np.cos(a), d*np.sin(a); xs.append(vx); ys.append(vy)
        capS.append(cap(vx, vy, [0], IMG)); capM.append(cap(vx, vy, offs, IMG))
    xs, ys = np.array(xs), np.array(ys); capS, capM = np.array(capS), np.array(capM)
    fig, ax = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
    def rings(A_):
        A_.add_patch(Circle((0,0), R_URB, color="cyan", fill=False, lw=2))
        A_.add_patch(Circle((0,0), R_RUR, color="magenta", fill=False, ls="--", lw=2))
        A_.add_patch(Circle((0,0), R_OUT, color="magenta", fill=False, ls="--", lw=1.5))
    def scat(A_, mask):
        A_.scatter(xs[mask],  ys[mask],  c=CAP_C,  s=2, alpha=0.6, linewidths=0)
        A_.scatter(xs[~mask], ys[~mask], c=MISS_C, s=2, alpha=0.6, linewidths=0)
    ax[0].set_title(f"Single patch approach\nCapture rate: {capS.mean():.1%}", fontweight="bold"); rings(ax[0])
    ax[0].add_patch(Rectangle((-IMG/2,-IMG/2), IMG, IMG, lw=3, edgecolor="black", facecolor="none"))
    scat(ax[0], capS)
    ax[1].set_title(f"MIL 3x3 approach\nCapture rate: {capM.mean():.1%}", fontweight="bold"); rings(ax[1])
    for dx in offs:
        for dy in offs:
            ax[1].add_patch(Rectangle((dx-IMG/2, dy-IMG/2), IMG, IMG, lw=1, edgecolor="blue", facecolor="blue", alpha=0.05))
    scat(ax[1], capM)
    for a in ax: a.set_xlim(-11,11); a.set_ylim(-11,11); a.set_aspect("equal"); a.set_xlabel("km"); a.grid(alpha=0.3)
    ax[0].set_ylabel("km")
    handles = [Line2D([0],[0], color="cyan", lw=2, label="Urban limit (2 km)"),
               Line2D([0],[0], color="magenta", lw=2, ls="--", label="Rural limit (5 km)"),
               Line2D([0],[0], color="magenta", lw=1.5, ls="--", label="Rural limit (10 km)"),
               Line2D([0],[0], color="black", lw=2, label="Tile coverage"),
               Line2D([0],[0], marker="o", color="w", markerfacecolor=CAP_C, markersize=8, label="Captured cluster"),
               Line2D([0],[0], marker="o", color="w", markerfacecolor=MISS_C, markersize=8, label="Missed cluster")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Coverage comparison: single patch vs MIL approach", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    return F.save(fig, "figure_coverage.png")

if __name__ == "__main__":
    figure_coverage()
