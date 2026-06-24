import os, urllib.request
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

from matplotlib import font_manager as _fm
import glob as _glob
for _fp in _glob.glob("/usr/share/fonts/truetype/crosextra/Carlito-*.ttf"):
    try: _fm.fontManager.addfont(_fp)
    except Exception: pass


TITLE, SUBTITLE, LABEL, TICK, LEGEND, ANNOT = 15, 13, 12, 11, 11, 11
def apply_style():
    plt.rcParams.update({
        "font.family": "Carlito",
        "font.size": ANNOT,
        "axes.titlesize": SUBTITLE, "axes.titleweight": "bold",
        "axes.labelsize": LABEL,
        "xtick.labelsize": TICK, "ytick.labelsize": TICK,
        "legend.fontsize": LEGEND, "legend.title_fontsize": LEGEND,
        "figure.titlesize": TITLE, "figure.titleweight": "bold",
        "axes.unicode_minus": False,
    })
apply_style()

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR       = str(PACKAGE_ROOT / "outputs" / "figures")
WORLD_GEOJSON = str(PACKAGE_ROOT / "data" / "boundaries" / "ne_110m_admin_0_countries.geojson")
NE_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
          "master/geojson/ne_110m_admin_0_countries.geojson")

C_MS, C_NL, C_FUSED, C_DARK, C_DIST = "#4C72B0", "#DD8452", "#55A868", "#2d7a49", "#4C72B0"
FOLD_COLOURS = {"A":"#1f77b4","B":"#ff7f0e","C":"#d62728","D":"#2ca02c","E":"#9467bd"}
PCT = {0.05:"5%",0.1:"10%",0.2:"20%",0.4:"40%",0.6:"60%",0.8:"80%",1.0:"100%"}

def out(name):
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, name)

def save(fig, name, also_pdf=False):
    p = out(name); fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    if also_pdf: fig.savefig(out(name.replace(".png", ".pdf")), bbox_inches="tight", facecolor="white")
    plt.close(fig); print(f"[SAVED] {name}"); return p

def basemap():
    import geopandas as gpd
    if not os.path.exists(WORLD_GEOJSON):
        print("fetching Natural Earth basemap ...")
        req = urllib.request.Request(NE_URL, headers={"User-Agent": "Mozilla/5.0"})
        open(WORLD_GEOJSON, "wb").write(urllib.request.urlopen(req, timeout=60).read())
    return gpd.read_file(WORLD_GEOJSON)

def bar_color(v):
    if v < 0: return "#d73027"
    if v < 0.3: return "#fc8d59"
    if v < 0.5: return "#fee090"
    if v < 0.7: return "#91cf60"
    return "#1a9850"

def hbar_panel(ax, cr, val_col, pooled_val, title, name_col="name", xlim=(-0.25, 1.05), fs=ANNOT):
    cr = cr.sort_values(val_col, ascending=True); y = np.arange(len(cr))
    ax.barh(y, cr[val_col].clip(lower=xlim[0]), height=0.6,
            color=[bar_color(v) for v in cr[val_col]], edgecolor="white", lw=0.5, zorder=3)
    ax.axvline(0, color="black", lw=0.8, zorder=4)
    ax.axvline(pooled_val, color=C_DARK, lw=1.5, ls="--", zorder=4, alpha=0.8,
               label=f"Pooled R\u00b2={pooled_val:.3f}")
    for i, (_, row) in enumerate(cr.iterrows()):
        v = row[val_col]; x = max(v, xlim[0])
        ax.text(x + (0.015 if v >= 0 else -0.015), i, f"{v:.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=fs, color="#222",
                fontweight="semibold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.75))
    ax.set_yticks(y); ax.set_yticklabels(cr[name_col], fontsize=TICK)
    ax.set_xlabel("Fused R\u00b2", fontsize=LABEL); ax.set_title(title, fontsize=SUBTITLE, fontweight="bold")
    ax.set_xlim(*xlim); ax.grid(True, axis="x", alpha=0.3, ls="--", zorder=0); ax.legend(fontsize=LEGEND, loc="lower right")

def smooth(x, y, n=300, clip_lo=None, k=3):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if clip_lo is not None: y = np.clip(y, clip_lo, None)
    m = np.isfinite(x) & np.isfinite(y); x, y = x[m], y[m]
    if len(x) < 3: return x, y
    k = min(k, len(x) - 1); xs = np.linspace(x[0], x[-1], n)
    return xs, make_interp_spline(x, y, k=k)(xs)
