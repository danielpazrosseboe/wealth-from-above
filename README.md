# Wealth from Above

**Estimating Household Asset Wealth Across 22 Sub-Saharan African Countries from Satellite Imagery, 2017–2024**

Replication code for the MSc Economics thesis of the same name (BI Norwegian Business School, 2026). The project fuses multispectral Landsat 8/9 imagery and VIIRS nighttime lights through a ResNet-18 convolutional network to predict cluster-level Demographic and Health Survey (DHS) asset wealth across 22 African countries, and benchmarks the result head-to-head against Meta's Relative Wealth Index on identical DHS clusters.

> **This repository contains code and rendered figures/tables only - no data.** Every input is obtained by the reproducer from its original source (see [DATA.md](DATA.md) and [`data/README.md`](data/README.md)). This keeps the repository fully compliant with the DHS Terms of Use and means results are reproduced from primary sources rather than from second-hand numbers.

---

## Headline results

| Configuration | Cluster fused R² | District (ADM2) fused R² |
|---|---|---|
| Full dataset, in-country CV | **0.682** | 0.812 |
| Full dataset, out-of-country CV | 0.638 | 0.734 |
| West Africa, single-patch | 0.587 | 0.715 |
| West Africa, MIL 3×3 | 0.612 | 0.751 |

Against the Relative Wealth Index on identical clusters, the model explains more within-country variation in **all 22 countries** (standardized R² 0.660 vs 0.530 in-country), with the per-country advantage significant in 20 of 22.

---

## What this repository contains

Code is organized one file per analysis stage, one file per thesis figure, and one file per thesis table, numbered to match the thesis (e.g. `figure_06_scatter.py` → Figure 6, `table_08…` → Table 8). The rendered figures and tables live in `reference_outputs/`.

```
.
├── code/                         # analysis, figure, and table scripts
│   ├── analysis_01_pca_wealth_index.py        # build the DHS PCA wealth index (the label)
│   ├── analysis_02a/b_export_imagery_*.py     # Google Earth Engine export (single-patch / MIL)
│   ├── analysis_03a/b_validation_and_folds_*  # cleaning, normalization, DBSCAN + OOC folds
│   ├── analysis_04a–g_train_*.py              # train MS & NTL branches (Full IC/OOC, West Africa SP/MIL)
│   ├── analysis_08_rwi_benchmark.py           # Relative Wealth Index head-to-head
│   ├── analysis_results.py                    # pooled metrics from the prediction files
│   ├── figure_01–12 / figure_A1,A2 / *_supp   # all main-body + appendix figures
│   ├── table_01–10 / table_A1 / *_supp        # all main-body + appendix tables
│   └── figure_common.py, table_common.py      # shared styling helpers
├── reference_outputs/            # the rendered thesis figures & tables (PNG) — one per figure_/table_ script
├── data/                         # NOT included — fetch every input yourself
│   └── README.md                 # what to download (DHS, RWI, geoBoundaries) and where to put it
├── PUBLISHING.md                 # how this repo was published + how to cut a citable release
├── requirements.txt
```

## Method in brief

- **Label.** A PCA asset-wealth index built from harmonized DHS asset and housing variables, aggregated to the survey cluster (16,279 clusters, 22 countries, 2017–2024).
- **Inputs.** Two complementary streams per cluster: seven-band Landsat 8/9 multispectral reflectance and VIIRS nighttime-light radiance, each a three-year median composite (2017–2019, 2020–2022, 2023–2025), exported via Google Earth Engine as 224×224 tiles (6.72 km).
- **Model.** A ResNet-18 (v2 pre-activation) backbone per stream; the 512-dim pooled features of each are concatenated and fused with ridge regression (1024-dim).
- **Evaluation.** Two cross-validation schemes - in-country (DBSCAN spatial declustering to block leakage) and out-of-country (whole nations held out) - plus a Multiple Instance Learning 3×3 variant on West Africa as a diagnostic of DHS coordinate displacement.

## Reproducing the results

The repository ships the full code and the rendered `reference_outputs/`, but **no data**, so reproduction is end-to-end from primary sources:

```bash
pip install -r requirements.txt

# 1. Obtain the inputs yourself (see data/README.md) and place them under data/:
#    - DHS surveys + GPS datasets   (registered access; restricted)
#    - Relative Wealth Index        (Humanitarian Data Exchange, CC BY 4.0)
#    - geoBoundaries ADM2           (CC BY 4.0)

cd code
# 2. Build the label, export imagery, build folds, train, predict.
#    (analysis_02*/04* additionally need TensorFlow, an Earth Engine project, and a GPU.)
python analysis_01_pca_wealth_index.py
python analysis_03a_validation_and_folds_singlepatch.py
python analysis_04a_train_full_incountry_ms.py
# ...run the analysis_* stages you need...

# 3. Render any figure/table from your regenerated predictions:
python figure_06_scatter.py            # -> ./figures_out/
python table_07_08_rwi_benchmark.py    # -> ./tables_out/
```

The values these reproduce: cluster fused R² 0.682 (in-country) / 0.638 (out-of-country), district 0.812 / 0.734. Each PNG in `reference_outputs/` is the target render for the matching `figure_*` / `table_*` script.

## Data and licensing

Code is released under the MIT License. **No data are distributed in this repository.** Each input keeps its own terms: DHS data are restricted (registered access; redistribution prohibited), the Relative Wealth Index and geoBoundaries are CC BY 4.0, and Landsat/VIIRS are public-domain US government data accessed through Google Earth Engine. See **[DATA.md](DATA.md)** and [`data/README.md`](data/README.md) to obtain them.


## Acknowledgements

Supervised by Dr. Simon Galle (BI Norwegian Business School). Built on the DHS Program, NASA/USGS Landsat, NOAA VIIRS, Meta's Relative Wealth Index, and geoBoundaries.
