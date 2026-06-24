# Obtaining the data

This repository ships **no data**. To reproduce the analysis, fetch each input
from its original source and place it under `data/` using the paths the scripts
expect. Nothing in this folder (except this file) is committed.

## What goes where

| Path | How to fill it |
|---|---|
| `data/metadata/` | Built by `code/analysis_01_pca_wealth_index.py` from the DHS surveys (cluster wealth index, GPS, urban flag). |
| `data/predictions/` | Produced by the training stages (`code/analysis_04*`). |
| `data/rwi/` | Relative Wealth Index CSVs, one per country (e.g. `nga_relative_wealth_index.csv`). |
| `data/boundaries/` | geoBoundaries ADM2 (`adm2_geoboundaries_combined.gpkg`). |
| `data/learning_curves/` | Produced by the West Africa training stages (`code/analysis_04e–g`). |

## Sources

- **DHS surveys + GPS datasets** — register at <https://dhsprogram.com/data/new-user-registration.cfm>, then request the relevant surveys and their GPS datasets. **Restricted; do not redistribute.**
- **Relative Wealth Index** — Humanitarian Data Exchange (Meta), CC BY 4.0.
- **geoBoundaries ADM2** — <https://www.geoboundaries.org>, CC BY 4.0.
- **Landsat 8/9 & VIIRS** — pulled live via Google Earth Engine (authenticated project) by `code/analysis_02*`.

See [`../DATA.md`](../DATA.md) for licenses and attribution lines.
