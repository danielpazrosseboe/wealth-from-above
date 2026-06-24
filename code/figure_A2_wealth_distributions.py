import pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import analysis_results as A, figure_common as F

def figure_wealth_distributions():
    cl = pd.read_csv(A.CONFIG.CLUSTERS_CSV)
    for by, fname, title in [("country", "figureA_wealth_by_country.png", "Distribution of pooled wealth index by country"),
                             ("year",    "figureA_wealth_by_year.png",    "Distribution of pooled wealth index by survey year")]:
        groups = sorted(cl[by].dropna().unique(), key=lambda v: str(v))
        data = [cl[cl[by] == g]["wealthpooled"].dropna().values for g in groups]
        fig, ax = plt.subplots(figsize=(12, 5))
        bp = ax.boxplot(data, labels=[str(g) for g in groups], showfliers=True, patch_artist=True)
        for b in bp["boxes"]: b.set_facecolor("#4C72B0"); b.set_alpha(0.8)
        for m in bp["medians"]: m.set_color("orange")
        ax.set_title(title, fontsize=12); ax.set_ylabel("wealthpooled"); ax.tick_params(axis="x", rotation=90, labelsize=7)
        ax.grid(True, axis="y", alpha=0.3); fig.tight_layout(); F.save(fig, fname)

if __name__ == "__main__":
    figure_wealth_distributions()
