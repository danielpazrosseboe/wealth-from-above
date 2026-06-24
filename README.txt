MASTER THESIS REPRODUCTION PACKAGE
No TFRecords version

This folder contains the files needed to reproduce the thesis analysis, tables,
and figures from saved prediction outputs. It intentionally excludes TFRecords
and raw CNN retraining inputs.


1. FOLDER STRUCTURE

code/
    Python scripts for the analysis, figures, and tables.

data/predictions/
    full_incountry_predictions.csv
    full_outofcountry_predictions.csv
    west_africa_singlepatch_predictions.csv
    west_africa_mil_predictions.csv

data/metadata/
    clusters_wealth_index.csv
    dhs_combined_metadata.csv

data/boundaries/
    adm2_geoboundaries_combined.gpkg

data/rwi/
    Relative Wealth Index CSVs for the 22 thesis countries.

data/learning_curves/
    west_africa_singlepatch_learning_curve.csv
    west_africa_mil_learning_curve.csv

reference_outputs/
    Rendered thesis figures and tables for comparison.

docs/
    Master_Thesis_REVISED_14.docx

outputs/
    Created automatically when scripts are run.


2. INSTALL REQUIREMENTS

Create or activate a Python environment, then install:

    pip install -r requirements.txt

The analysis scripts mainly need:

    numpy
    pandas
    scipy
    scikit-learn
    matplotlib
    geopandas
    pyreadstat

Notes:

    - scipy is used for the RWI benchmark and sign tests.
    - geopandas is used for ADM2 district aggregation and map figures.
    - matplotlib is used to render figure and table PNGs.


3. REPRODUCE THE MAIN ANALYSIS

From this package folder, run:

    cd code
    python analysis_results.py

This writes analysis CSVs to:

    outputs/analysis/

Expected headline values:

    Full in-country fused R^2:        0.6819
    Full out-of-country fused R^2:    0.6383
    West Africa single-patch fused R^2: 0.5870
    West Africa MIL fused R^2:          0.6120


4. REPRODUCE TABLES

From the code folder:

    python table_01_decomposition.py
    python table_02_perfold_spread.py
    python table_03_country_full.py
    python table_04_country_wa.py
    python table_05_district.py
    python table_06_error_structure.py
    python table_07_08_rwi_benchmark.py
    python table_09_temporal_robustness.py
    python table_10_urban_rural_mil.py
    python table_A1_pc1_loadings.py

Generated table PNGs are written to:

    outputs/tables/

Tables 9 and 10 render the thesis-reported values directly. This matches the
thesis code design because their exact sample construction depends on
intermediate artifacts that are not part of this no-TFRecords package.


5. REPRODUCE FIGURES

From the code folder:

    python figure_01_coverage.py
    python figure_02_fold_maps.py
    python figure_04_country_bars_full.py
    python figure_05_country_bars_wa.py
    python figure_06_scatter.py
    python figure_07_africa_maps.py
    python figure_08_district_bars.py
    python figure_09_residual_bias.py
    python figure_10_12_rwi_benchmark.py
    python figure_A1_cluster_distribution.py
    python figure_A2_wealth_distributions.py

Generated figure PNGs are written to:

    outputs/figures/

Map figures may download a Natural Earth basemap on first run if
data/boundaries/ne_110m_admin_0_countries.geojson is not already present.


6. FIGURE 3 NOTE

The rendered thesis version of Figure 3 is included here:

    reference_outputs/Figure_03_learning_curves.png

The two full-dataset learning-curve CSVs were not found in Downloads, so Figure
3 cannot be fully regenerated from this package yet. The missing files are:

    data/learning_curves/full_incountry_learning_curve.csv
    data/learning_curves/full_outofcountry_learning_curve.csv

The West Africa learning-curve files are present.


7. RWI BENCHMARK

The RWI files are in:

    data/rwi/

To reproduce the RWI comparison figures and tables:

    cd code
    python table_07_08_rwi_benchmark.py
    python figure_10_12_rwi_benchmark.py


8. DISTRICT AGGREGATION

District aggregation uses:

    data/metadata/dhs_combined_metadata.csv
    data/boundaries/adm2_geoboundaries_combined.gpkg

Run:

    cd code
    python table_05_district.py
    python figure_08_district_bars.py

This requires geopandas.


9. WHAT IS NOT INCLUDED

This package does not include:

    - TFRecords
    - raw satellite image exports
    - CNN checkpoint files
    - end-to-end retraining inputs

It is meant for reproducing the thesis results, figures, and tables from saved
prediction outputs, not for retraining the CNN models from raw imagery.

