import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import figure_common as _F
TITLE_SZ, HEAD_SZ, CELL_SZ = 14, 11, 11

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = str(PACKAGE_ROOT / "outputs" / "tables")

def out(name):
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, name)

def fmt(v):
    try:
        if v is None or np.isnan(float(v)): return "\u2014"
    except Exception: return "\u2014"
    return f"{float(v):.3f}".replace("-", "\u2212")

def fmtd(v):
    try:
        if v is None or np.isnan(float(v)): return "\u2014"
    except Exception: return "\u2014"
    return f"{float(v):+.3f}".replace("-", "\u2212")

def academic_table_png(title, headers, rows, col_align, out_path, bold_last=True,
                        group_after=None, figw=11, rowh=0.32, label_frac=0.20):
    group_after = group_after or set()
    n_rows, n_cols = len(rows), len(headers)
    fig, ax = plt.subplots(figsize=(figw, 1.1 + rowh*(n_rows+1)))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    w0 = label_frac; rest = (1.0 - w0) / (n_cols - 1)
    edges = [0.0, w0] + [w0 + rest*j for j in range(1, n_cols)]
    bc = lambda j: (edges[j] + edges[j+1]) / 2.0
    bl = lambda j: edges[j] + 0.008
    br = lambda j: edges[j+1] - 0.008
    total = n_rows + 1; yh = 1.0 / (total + 0.8); yof = lambda r: 1.0 - (r + 0.8) * yh
    ax.plot([0,1], [yof(-0.45)]*2, color="black", lw=1.6)
    ax.plot([0,1], [yof(0.55)]*2,  color="black", lw=0.9)
    ax.plot([0,1], [yof(total-0.45)]*2, color="black", lw=1.6)
    for gi in group_after: ax.plot([0,1], [yof(gi+0.55)]*2, color="black", lw=0.5, alpha=0.55)
    def place(j, text, y, weight):
        a = col_align[j]
        if a == "l": ax.text(bl(j), y, text, ha="left", va="center", fontsize=CELL_SZ, fontweight=weight)
        elif a == "r": ax.text(br(j), y, text, ha="right", va="center", fontsize=CELL_SZ, fontweight=weight)
        else: ax.text(bc(j), y, text, ha="center", va="center", fontsize=CELL_SZ, fontweight=weight)
    for j, h in enumerate(headers):
        if col_align[j] == "l": ax.text(bl(j), yof(0), h, ha="left", va="center", fontsize=HEAD_SZ, fontweight="bold")
        else: ax.text(bc(j), yof(0), h, ha="center", va="center", fontsize=HEAD_SZ, fontweight="bold")
    for i, row in enumerate(rows):
        fw = "bold" if (bold_last and i == n_rows-1) else "normal"
        for j, v in enumerate(row): place(j, str(v), yof(i+1), fw)
    ax.set_title(title, fontsize=TITLE_SZ, fontweight="bold", pad=14)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[SAVED] {os.path.basename(out_path)}"); return out_path

def _r2_color(v):
    try: v = float(v)
    except Exception: return "#F0F0F0"
    if v >= 0.70: return "#a1d99b"
    if v >= 0.50: return "#ffffb2"
    if v >= 0.30: return "#fdae6b"
    if v >= 0.00: return "#fee0d2"
    return "#fc8d59"

def _delta_color(v):
    try: v = float(v)
    except Exception: return "#555555"
    return "#1a9850" if v > 0.02 else ("#d73027" if v < -0.02 else "#555555")

def styled_table_png(title, headers, rows, r2_cols, out_path, delta_col=None, figsize=(13, 8), fontsize=CELL_SZ):
    fig, ax = plt.subplots(figsize=figsize, facecolor="white"); ax.axis("off"); n = len(rows)
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center", bbox=[0,0,1,1])
    table.auto_set_font_size(False); table.set_fontsize(fontsize)
    for c in range(len(headers)):
        cell = table[0, c]; cell.set_facecolor("#1B3A5C"); cell.get_text().set_color("white")
        cell.get_text().set_weight("bold"); cell.set_edgecolor("white"); cell.set_height(0.09)
    for rr in range(1, n+1):
        last = (rr == n)
        for c in range(len(headers)):
            cell = table[rr, c]; val = rows[rr-1][c]; cell.set_edgecolor("#CCCCCC"); cell.set_height(0.063)
            if c == 0:
                cell.set_facecolor("#D0E8F8" if last else ("#FFFFFF" if rr % 2 else "#F5F5F5"))
                cell.get_text().set_weight("bold" if last else "normal")
            elif c in r2_cols:
                cell.set_facecolor(_r2_color(val))
                if last: cell.get_text().set_weight("bold")
            elif delta_col and c == delta_col:
                cell.set_facecolor("#D0E8F8" if last else ("#FFFFFF" if rr % 2 else "#F5F5F5"))
                cell.get_text().set_color(_delta_color(val)); cell.get_text().set_weight("bold")
            else:
                cell.set_facecolor("#D0E8F8" if last else ("#FFFFFF" if rr % 2 else "#F5F5F5"))
    ax.set_title(title, fontsize=TITLE_SZ, fontweight="bold", pad=16)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[SAVED] {os.path.basename(out_path)}"); return out_path

def booktabs_png(title, headers, rows, col_align, out_path, figw=11.0, aspect=2.5,
                 fontsize=12.5, bold_cols=None, edges=None):
    bold_cols = bold_cols or set()
    fig, ax = plt.subplots(figsize=(figw, figw / aspect))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    n_cols, n_rows = len(headers), len(rows)
    edges = np.asarray(edges) if edges is not None else np.linspace(0, 1, n_cols + 1)
    edges[0] = 0.0
    cx = lambda j: (edges[j] + edges[j + 1]) / 2
    lx = lambda j: edges[j] + 0.008
    top = 0.86
    header_h = 0.16 if any("\n" in h for h in headers) else 0.10
    row_h = (top - header_h - 0.05) / max(n_rows, 1)
    y_top, y_mid = top, top - header_h
    y_bot = y_mid - n_rows * row_h
    for yy, lw in [(y_top, 1.8), (y_mid, 1.0), (y_bot, 1.8)]:
        ax.plot([0, 1], [yy, yy], color="black", lw=lw, solid_capstyle="butt")
    yh = y_mid + header_h / 2
    for j, h in enumerate(headers):
        ax.text(cx(j) if col_align[j] == "c" else lx(j), yh, h,
                ha="center" if col_align[j] == "c" else "left", va="center",
                fontsize=HEAD_SZ, fontweight="bold", linespacing=1.4)
    for i, row in enumerate(rows):
        yr = y_mid - (i + 0.5) * row_h
        for j, val in enumerate(row):
            ax.text(cx(j) if col_align[j] == "c" else lx(j), yr, val,
                    ha="center" if col_align[j] == "c" else "left", va="center",
                    fontsize=CELL_SZ, fontweight="bold" if j in bold_cols else "normal")
    ax.set_title(title, fontsize=TITLE_SZ, fontweight="bold", pad=12)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print(f"[SAVED] {os.path.basename(out_path)}"); return out_path
