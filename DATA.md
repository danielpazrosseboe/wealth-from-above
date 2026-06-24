# Data sources, licensing, and access

This project uses four data sources with **different licenses**. **No data are included in this repository** — every input is fetched by you from its original source and placed under `data/` (see [`data/README.md`](data/README.md)). The code is MIT-licensed; the data are not.

## Summary

| Source | What it is | License / status |
|---|---|---|---|
| **DHS** (Demographic and Health Surveys) | Household survey microdata → the PCA wealth label, cluster GPS, urban/rural flag | **Restricted.** Registered access only; redistribution prohibited |
| **Landsat 8/9** (NASA/USGS) | Multispectral daytime reflectance | Public domain (US government) |
| **VIIRS** (NOAA) | Nighttime light radiance | Public domain (US government) | 
| **Relative Wealth Index** (Meta) | Benchmark wealth product, 2.4 km grid | **CC BY 4.0** (via Humanitarian Data Exchange) 
| **geoBoundaries** ADM2 | District boundaries for aggregation | **CC BY 4.0** | 

## DHS - restricted

The DHS Program Terms of Use state that **DHS micro-level data may not be re-distributed**, that datasets may not be shared with other researchers without written consent, and that users must **make no effort to identify any individual, household, or enumeration area**. (See *The DHS Program — Datasets Terms of Use*.)

To obtain the inputs: register at [dhsprogram.com](https://dhsprogram.com/data/new-user-registration.cfm), request the surveys and their **GPS datasets**, then run `code/analysis_01_pca_wealth_index.py` to rebuild the wealth index and cluster file locally. DHS also asks that you email a copy of resulting publications to `references@dhsprogram.com`.

## Relative Wealth Index and geoBoundaries - CC BY 4.0, fetch and attribute

Both are openly licensed but must be downloaded from source and attributed if you redistribute derivatives:

- **Relative Wealth Index:** Chi, G., Fang, H., Chatterjee, S., & Blumenstock, J. E. (2022), *Microestimates of wealth for all low- and middle-income countries*, PNAS; data via Meta / Humanitarian Data Exchange, CC BY 4.0.
- **geoBoundaries:** Runfola, D. et al. (2020), *geoBoundaries: A global database of political administrative boundaries*, PLoS ONE; CC BY 4.0.

## Landsat / VIIRS - public domain, via Earth Engine

Accessed live through Google Earth Engine by `code/analysis_02*`; no local copy is stored. Requires an authenticated Earth Engine project.

## The .gitignore already enforces this

Everything under `data/` (and any `*.csv`) is excluded except `data/README.md`. Before your first commit, confirm:

```bash
git add -A
git ls-files | grep -E '^data/|\.csv$' | grep -v '^data/README.md$' || echo "clean — no data staged"
```

Git history is permanent, so never let data into even the first commit.

## Sources

- [The DHS Program — Datasets Terms of Use](https://dhsprogram.com/data/terms-of-use.cfm)
- [The DHS Program — Request access to datasets](https://dhsprogram.com/data/new-user-registration.cfm)
