import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import analysis_08_rwi_benchmark as B
import table_common as T


def _booktabs(path, title, headers, rows, col_align, edges, aspect=2.5,
              fontsize=12.5, highlight_rows=None, bold_cols=None):
    highlight_rows = highlight_rows or set()
    bold_cols = bold_cols or set()
    figw = 11.0
    fig, ax = plt.subplots(figsize=(figw, figw / aspect))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    nrows = len(rows)
    edges = np.asarray(edges)

    def cx(j):
        return (edges[j] + edges[j + 1]) / 2

    def lx(j):
        return edges[j] + 0.008

    top = 0.86
    header_h = 0.16 if any("\n" in h for h in headers) else 0.10
    row_h = (top - header_h - 0.05) / nrows
    y_top, y_mid = top, top - header_h
    y_bot = y_mid - nrows * row_h
    for yy, lw in [(y_top, 1.8), (y_mid, 1.0), (y_bot, 1.8)]:
        ax.plot([0, 1], [yy, yy], color="black", lw=lw, solid_capstyle="butt")
    yh = y_mid + header_h / 2
    for j, h in enumerate(headers):
        ax.text(cx(j) if col_align[j] == "c" else lx(j), yh, h,
                ha="center" if col_align[j] == "c" else "left", va="center",
                fontsize=fontsize - 1.5, fontweight="bold", linespacing=1.4)
    for i, row in enumerate(rows):
        yr = y_mid - (i + 0.5) * row_h
        if i in highlight_rows:
            ax.add_patch(plt.Rectangle((0, yr - row_h / 2), 1, row_h,
                                       facecolor="#eef4fa", edgecolor="none", zorder=0))
        for j, val in enumerate(row):
            ax.text(cx(j) if col_align[j] == "c" else lx(j), yr, val,
                    ha="center" if col_align[j] == "c" else "left", va="center",
                    fontsize=fontsize - 1, fontweight="bold" if j in bold_cols else "normal")
    ax.set_title(title, fontsize=fontsize + 2, fontweight="bold", pad=12)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {path}")


def table_literature(out_path):
    headers = ["Study", "Coverage", "Imagery / data", "Outcome", "Resolution", "Held-out R\u00b2"]
    rows = [
        ["Jean et al. (2016)", "5 SSA", "Daytime + lights", "Consumption", "Cluster", "Up to 0.75"],
        ["Yeh et al. (2020)", "23 African", "Landsat + VIIRS", "DHS assets", "~6.7 km", "~0.70 (OOC)"],
        ["Chi et al. (2022)", "135 LMICs", "Sat. + connectivity", "Rel. wealth", "2.4 km", "~0.49 (this data)"],
        ["This thesis, SP", "22 African", "Landsat 8/9 + VIIRS", "DHS PCA", "6.72 km", "0.682 / 0.638"],
        ["This thesis, MIL", "11 W. Afr.", "+ 3\u00d73 patch bag", "DHS PCA", "~10.3 km", "0.612 (IC)"],
    ]
    _booktabs(out_path, "Benchmarking Against Published Wealth-Prediction Models",
              headers, rows, ["l"] * 6,
              edges=[0, 0.155, 0.275, 0.435, 0.585, 0.745, 1.0],
              aspect=2.5, fontsize=12)


def table_headtohead(agg, per, out_path):
    headers = ["Configuration", "n clusters", "Spearman \u03c1\n(fused model)", "Spearman \u03c1\n(RWI)",
               "Std. R\u00b2\n(fused model)", "Std. R\u00b2\n(RWI)"]
    labels = {"ic": "Full-IC", "ooc": "Full-OOC", "sp": "WA-SP", "mil": "WA-MIL"}
    rows = []
    for k in ["ic", "ooc", "sp", "mil"]:
        a = agg[k]
        n = int(per[k]["n"].sum())
        rows.append([labels[k], f"{n:,}", f"{a['m_rho']:.3f}", f"{a['r_rho']:.3f}",
                     f"{a['m_r2']:.3f}", f"{a['r_r2']:.3f}"])
    _booktabs(out_path, "Head-to-Head Comparison Against the Relative Wealth Index",
              headers, rows, ["l", "c", "c", "c", "c", "c"],
              edges=[0, 0.26, 0.40, 0.565, 0.69, 0.85, 1.0],
              aspect=2.45, fontsize=12.5, bold_cols={2, 4})


if __name__ == "__main__":
    per, agg, raw = B.run()
    table_literature(T.out("table_rwi_literature.png"))
    table_headtohead(agg, per, T.out("table_rwi_headtohead.png"))
