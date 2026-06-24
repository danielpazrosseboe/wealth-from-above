import pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import analysis_results as A, figure_common as F

def figure_cluster_distribution():
    import geopandas as gpd
    cl = pd.read_csv(A.CONFIG.CLUSTERS_CSV, usecols=["country","year","cluster","lat","lon"]).dropna(subset=["lat","lon"])
    gdf = gpd.GeoDataFrame(cl, geometry=gpd.points_from_xy(cl["lon"], cl["lat"]), crs="EPSG:4326")
    africa = F.basemap().cx[-25:60, -40:40]
    fig, ax = plt.subplots(figsize=(10, 10)); africa.boundary.plot(ax=ax, linewidth=0.5)
    gdf.plot(ax=ax, markersize=0.08, alpha=0.6)
    ax.set_title("DHS cluster distribution over Africa", fontsize=14)
    ax.set_xlabel("Longitude", fontsize=12); ax.set_ylabel("Latitude", fontsize=12); ax.set_aspect("equal", adjustable="box")
    plt.tight_layout(); return F.save(fig, "figureA_cluster_distribution.png")

if __name__ == "__main__":
    figure_cluster_distribution()
