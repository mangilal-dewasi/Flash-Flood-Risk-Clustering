# 🌧️ Flash-Flood Pattern Atlas

### Exploratory clustering of rainfall–terrain patterns across Himalayan catchments

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/) [![Method](https://img.shields.io/badge/Method-Unsupervised%20Learning-0F766E)](#methodology)

**Unsupervised Learning Research Project**  
**Study region:** Himachal Pradesh and Uttarakhand, India  
**Rainfall period:** 2019–2024  
**Current analysis table:** 113 catchments × 9 selected features

The **Flash-Flood Pattern Atlas** is an exploratory research and visualization project. It combines IMD daily gridded rainfall, terrain characteristics derived from SRTM elevation data, and HydroBASINS Level 8 catchment boundaries. Unsupervised-learning methods group catchments with similar rainfall–terrain profiles and support inspection of those patterns spatially.

> **Research boundary:** These are exploratory rainfall–terrain pattern groups—not validated flood-risk classes, flood forecasts, or estimates of flood probability. Independent observed flood-event labels were not used to train or validate the clusters. Do not use this project for emergency, warning, or site-specific safety decisions.

---

## Contents

- [Project at a glance](#project-at-a-glance)
- [Research questions](#research-questions)
- [Data sources and coverage](#data-sources-and-coverage)
- [Features used](#features-used)
- [Reproducible pipeline](#reproducible-pipeline)
- [Methodology](#methodology)
- [Current analysis results](#current-analysis-results)
- [Interactive dashboard](#interactive-dashboard)
- [Repository structure](#repository-structure)
- [Installation and execution](#installation-and-execution)
- [Generated outputs](#generated-outputs)
- [Reproducibility and data handling](#reproducibility-and-data-handling)
- [Limitations and next steps](#limitations-and-next-steps)
- [References and acknowledgements](#references-and-acknowledgements)

## Project at a glance

| Item | Description |
|---|---|
| Goal | Explore groups of catchments with similar rainfall and terrain characteristics |
| Study area | Selected Himalayan catchments in Himachal Pradesh and Uttarakhand, India |
| Rainfall period | IMD daily rainfall, 2019–2024 |
| Catchment framework | HydroBASINS Level 8; catchment selection applies the study-area boundary and a minimum-area criterion of 50 km² |
| Final clustering table | 113 catchments and 9 selected numeric features |
| Main method | Standardized-feature K-Means |
| Candidate solutions | k = 5 and k = 6 retained for comparison |
| Supplementary method | K-Medoids (PAM-style) using Manhattan distance |
| Validation | Silhouette, Davies–Bouldin, cluster-size diagnostics, and random-seed stability |
| Visualization | PCA projection, catchment map, cluster profiles, and Streamlit dashboard |

## Research questions

1. Do catchments in the study area form distinguishable groups based on rainfall and terrain characteristics?
2. How do candidate cluster counts compare using internal validation measures and repeated random initializations?
3. How do cluster profiles differ in the original rainfall and terrain units?
4. How do Euclidean K-Means and Manhattan-distance K-Medoids compare as exploratory grouping approaches?
5. How can the patterns be mapped without presenting cluster membership as a measured flood outcome?

## Data sources and coverage

| Dataset | Use | Source |
|---|---|---|
| IMD daily gridded rainfall, 0.25° × 0.25°, annual NetCDF files for 2019–2024 | Catchment rainfall summaries and heavy-rainfall indicators | [IMD rainfall dataset](https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html) |
| SRTM 1 Arc-Second Global elevation tiles | Elevation, relief, and slope summaries | [USGS SRTM documentation](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm-1/) |
| HydroBASINS Level 8 polygons | Catchment boundaries and identifiers | [HydroBASINS / HydroSHEDS](https://www.hydrosheds.org/products/hydrobasins) |
| Study boundary layer | Geographic selection of the study region; not a clustering feature | Project input boundary data |

The pipeline preflight reported six annual IMD NetCDF files (2019–2024) and 42 SRTM HGT tiles in the local input folders. These are input availability checks; they do not by themselves establish that every source pixel is valid or complete.

The rainfall and terrain feature tables each contain 115 catchment records. Two catchments were excluded from the merged/modeling table because their rainfall spatial-coverage fractions did not meet the implemented coverage rule, despite complete temporal coverage. The resulting modeling/clustering table contains 113 catchments. The exclusion details are recorded in `rainfall_coverage_exclusions.csv`; document the exact spatial-coverage cutoff from the preprocessing code when reporting the method.

Large source datasets—especially annual NetCDF files and SRTM tiles—should generally remain outside ordinary Git commits. Keep scripts, source links, processing instructions, and appropriate derived outputs in the repository. Follow the data providers’ current attribution and usage terms.

## Features used

The final clustering dataset uses these nine features:

| Feature | Interpretation |
|---|---|
| `mean_monsoon_rainfall_mm` | Mean monsoon rainfall characteristic for a catchment |
| `mean_heavy_rain_days_50mm` | Mean number of days meeting the 50 mm heavy-rain threshold |
| `mean_very_heavy_rain_days_100mm` | Mean number of days meeting the 100 mm very-heavy-rain threshold |
| `mean_max_3day_rainfall_mm` | Mean maximum three-day rainfall metric |
| `rainfall_cv` | Rainfall coefficient of variation |
| `mean_elevation_m` | Mean catchment elevation in metres |
| `relief_m` | Catchment relief in metres |
| `mean_slope_deg` | Mean slope in degrees |
| `slope_std_deg` | Standard deviation of slope in degrees |

Rainfall summaries are derived from the 2019–2024 daily record. The extraction workflow defines monsoon months as June–September and uses daily rainfall thresholds of 50 mm and 100 mm for heavy and very-heavy rainfall indicators. The clustering feature table is standardized before distance-based clustering because the features have different measurement units and scales.

## Reproducible pipeline

The project includes a pipeline entry point, `pipeline.py`, intended to orchestrate the preprocessing and final analysis scripts in sequence. Run it from the repository root so relative file paths resolve correctly.

### Pipeline stages

1. **Preflight checks:** check the required HydroBASINS Level 8 shapefile, study boundary, six annual rainfall NetCDF files, 42 SRTM HGT tiles, and expected source scripts.
2. **Prepare rainfall input:** combine annual IMD rainfall files into the pipeline’s interim multi-year NetCDF.
3. **Prepare catchments:** create the study-region catchment set from HydroBASINS and the study boundary; apply the 50 km² catchment-area filter.
4. **Extract rainfall features:** calculate catchment-level rainfall indicators and coverage information.
5. **Extract terrain features:** derive terrain summaries from the SRTM tiles.
6. **Build the modeling dataset:** merge rainfall and terrain tables by catchment identifier and apply coverage/valid-row checks.
7. **Select clustering features:** create the nine-feature clustering dataset.
8. **Run final analysis:** evaluate K-Means candidates, compare K-Means with Manhattan-distance K-Medoids, produce assignments/profiles, inspect seed stability, and export PCA diagnostics and related tables.

`create_final_catchments.py` is a preprocessing step in the pipeline: it creates the catchment boundary GeoPackage used by later feature extraction and mapping. It should be called by `pipeline.py`; merely storing the script in the repository does not execute it.



## Methodology

1. **Catchment preparation.** Select HydroBASINS Level 8 catchments within the study boundary and apply the minimum catchment-area criterion.
2. **Rainfall feature extraction.** Derive catchment-level statistics from IMD daily gridded rainfall. The workflow uses area-weighted aggregation across intersecting grid cells.
3. **Terrain feature extraction.** Use SRTM elevation data to derive catchment elevation, relief, and slope summaries.
4. **Dataset assembly and coverage screening.** Join rainfall and terrain features by catchment ID, check spatial coverage, and retain valid rows. Keep catchment IDs as strings to avoid accidental numeric conversion.
5. **Feature selection and scaling.** Use the nine documented features and standardize them before clustering.
6. **K-Means evaluation.** Evaluate candidate cluster counts from k = 2 through k = 8 with internal metrics, including silhouette, Davies–Bouldin, and Calinski–Harabasz diagnostics.
7. **Candidate comparison and stability.** Retain k = 5 and k = 6 for interpretation and assess consistency across tested random seeds using adjusted Rand index (ARI).
8. **Distance-method comparison.** Compare Euclidean K-Means with a PAM-style K-Medoids implementation using Manhattan distance. The medoid procedure uses heuristic initialization and improving swaps; it is not a guarantee of a global optimum.
9. **PCA exploration.** Use a two-component PCA projection for visualization and report explained variance/loadings. PCA is not the space in which the primary K-Means solution is fitted; clustering uses all nine standardized features.
10. **Dashboard exploration.** Inspect assignments, cluster sizes, original-unit and standardized profiles, map patterns, and validation outputs.

**Metric interpretation:** K-Means inertia and K-Medoids’ sum of Manhattan distances to medoids are different objectives. Their raw objective values must not be compared numerically as though they measured the same thing. The comparison report uses each method’s stated distance for silhouette; Davies–Bouldin and Calinski–Harabasz are Euclidean diagnostics.

## Current analysis results

The current consolidated output reports 113 catchments. Results below are from the current generated analysis and may change if source data, preprocessing, features, or code changes.

| Diagnostic | k = 5 | k = 6 |
|---|---:|---:|
| Silhouette score | **0.3257** | 0.3235 |
| Davies–Bouldin index | **1.0270** | 1.0275 |
| Mean pairwise ARI across 10 seeds | 0.9110 | **0.9913** |
| Minimum pairwise ARI across 10 seeds | 0.7492 | **0.9809** |
| Smallest cluster size | 9 | 7 |

The current evidence is mixed: **k = 5 has slightly stronger internal separation scores; k = 6 has much higher consistency across the tested random seeds.** Keep both as candidates and explain the selection rationale. Stability across seeds does not prove that clusters are physically meaningful or correspond to flood occurrence.

The first two PCA components explain approximately **73.72%** of the standardized feature variance (PC1: 50.14%; PC2: 23.58%). A 2D projection is useful for inspection but cannot preserve all structure in the full nine-dimensional feature space.

For the Euclidean K-Means vs. Manhattan K-Medoids comparison, use `final_distance_method_comparison.csv` and the associated assignment/medoid files. Interpret silhouette scores using the relevant method distance and do not compare method-specific objective values directly.

## Interactive dashboard

The Streamlit dashboard provides:

- **Overview** — study snapshot, catchment count, selected solution, cluster counts, and interpretation caveats.
- **Catchment map** — catchments colored by exploratory cluster membership when the catchment GeoPackage is available.
- **Cluster profiles** — summaries in original units and standardized profile views.
- **PCA explorer** — two-component projection and component loadings.
- **Validation & methods** — K-Means evaluation, seed stability, and available distance-method comparison.
- **Dataset explorer** — inspect and download processed catchment-level data and assignments.

The project’s Streamlit header/toolbar is intentionally hidden in the current UI styling.

## Repository structure

The intended repository layout is organized by pipeline entry points, source scripts, and data stages. Match this structure to the actual files in your checkout; filenames must match the paths invoked by `pipeline.py`.

```text
Flash-Flood-Risk-Clustering/
├── app.py
├── download.py
├── pipeline.py
├── requirements.txt
├── README.md
├── src/
│   ├── create_final_catchments.py
│   ├── extract_rainfall_features.py
│   ├── extract_terrain_features.py
│   ├── build_modeling_dataset.py
│   ├── select_clustering_features.py
│   └── final_project_analysis_pca_manhattan.py
├── data/
│   ├── hydrobasins/       # HydroBASINS Level 8 source files
│   ├── raw/               # downloaded source data, including SRTM tiles
│   ├── interim/           # study boundary and prepared intermediate inputs
│   └── processed/         # feature tables, clustering outputs, and profiles
└── reports/
    └── figures/           # generated analysis figures, if retained
```

Diagnostic scripts and exploratory notebooks (for example, catchment-size checks, correlation checks, SRTM tile listing, and rainfall data exploration) are useful for research and troubleshooting, but are not automatically executed unless `pipeline.py` explicitly calls them. An optional NASA Earthdata SRTM downloader is an alternate acquisition route; do not run it redundantly when the project’s `download.py` already obtains the required SRTM inputs.

## Installation and execution

Use a Python environment compatible with the dependencies listed in the repository’s `requirements-dashboard.txt`. Geospatial libraries can have platform-specific installation requirements; use the project dependency file rather than installing packages ad hoc.

### Windows PowerShell

```powershell
cd "C:\Users\mangi\Desktop\INT396 Project"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dashboard.txt

# Optional: acquire/extract inputs using the project's download script
python download.py

# Verify inputs and expected scripts only
python pipeline.py --check-only

# Execute the full preprocessing and analysis pipeline
python pipeline.py

# Launch the dashboard after the pipeline has completed
streamlit run streamlit_app.py
```

## Generated outputs

The pipeline writes its CSV outputs under `data/processed/`. The current generated set includes the following core outputs (some are optional diagnostics or analysis products):

| Output | Purpose |
|---|---|
| `catchment_rainfall_features.csv` | Catchment rainfall feature table |
| `catchment_terrain_features.csv` | Catchment terrain feature table |
| `catchment_features_merged_all.csv` | Joined rainfall–terrain table before final screening |
| `modeling_dataset.csv` | Valid merged dataset used for modeling |
| `clustering_dataset.csv` | Final nine-feature clustering input |
| `rainfall_coverage_exclusions.csv` | Catchments excluded by rainfall coverage screening |
| `feature_summary.csv` / `final_feature_summary.csv` | Feature summary diagnostics |
| `feature_correlation.csv` / `final_feature_correlation.csv` | Feature correlation diagnostics |
| `clustering_feature_correlation.csv` | Correlation diagnostics for the selected clustering features |
| `final_kmeans_evaluation.csv` | K-Means candidate metrics and cluster-size diagnostics |
| `final_k5_assignments.csv`, `final_k6_assignments.csv` | Catchment cluster labels for k = 5 and k = 6 |
| `final_k5_profiles_original_units.csv`, `final_k6_profiles_original_units.csv` | Cluster profiles in original feature units |
| `final_k5_profiles_zscores.csv`, `final_k6_profiles_zscores.csv` | Standardized cluster profiles |
| `final_k5_silhouette_by_cluster.csv`, `final_k6_silhouette_by_cluster.csv` | Cluster-level silhouette summaries |
| `final_distance_method_comparison.csv` | K-Means Euclidean vs. K-Medoids Manhattan comparison |
| `final_distance_method_assignments.csv` | Assignments for the distance-method comparison |
| `final_kmedoids_manhattan_k5_medoids.csv`, `final_kmedoids_manhattan_k6_medoids.csv` | Selected medoids for the Manhattan K-Medoids solutions |
| `final_seed_metrics.csv`, `final_seed_stability_summary.csv` | Per-seed metrics and stability summaries |
| `final_pca_explained_variance.csv`, `final_pca_loadings.csv` | PCA variance and loading diagnostics |
| `final_feature_set_sensitivity.csv` | Feature-set sensitivity check, when generated |
| `final_high_feature_correlations.csv` | High-correlation feature pairs, when generated |

The catchment boundary GeoPackage is used by the map and should be present at the path expected by the dashboard (currently `data/processed/final_catchments_L8_50km2.gpkg`). The exact output list can vary by pipeline version; check the script’s output definitions before assuming an optional file will be generated.

## Reproducibility and data handling

- Run commands from the repository root.
- `python pipeline.py --check-only` validates inputs and expected scripts; it does not generate the processed tables.
- `python pipeline.py` is the full processing command.
- Keep the rainfall window (2019–2024), monsoon definition (June–September), rainfall thresholds (50 mm and 100 mm), catchment-area criterion, and coverage rule consistent when reproducing the current outputs.
- Preserve catchment IDs as strings when reading, joining, or writing CSVs.
- Record the script version, parameters, feature set, random seeds, and source data versions for each final run.
- Avoid committing large raw NetCDF and SRTM files unless repository storage and provider terms allow it. Use clear instructions for acquiring source data.
- Keep generated outputs separate from source inputs and intermediate files.
- Pipeline preflight passing confirms required inputs/scripts were found; it does not prove that the complete workflow ran or that the analysis is scientifically validated.
- Do not compare objective values from algorithms that optimize different functions directly.
- Keep optional diagnostics/notebooks separate from the core automated workflow unless they are explicitly wired into `pipeline.py`.

## Limitations and next steps

### Current limitations

- **No independent observed flood labels:** this is unsupervised pattern discovery, not supervised flood prediction or validated risk classification.
- **Moderate separation:** silhouette scores around 0.32 suggest meaningful but not cleanly separated groups; inspect overlap and cluster-level diagnostics.
- **Small clusters:** the k = 6 solution includes a cluster of only 7 catchments. Interpret small groups cautiously and examine their feature profiles and geography.
- **Feature dependence:** rainfall indicators may be correlated, meaning related rainfall information can influence clustering more than once.
- **Spatial and temporal scope:** findings apply to the selected catchments and 2019–2024 rainfall period; they should not automatically be generalized to other regions or periods.
- **Input uncertainty:** gridded rainfall resolution, spatial coverage, DEM resolution, and terrain derivation choices can affect catchment summaries.
- **Method uncertainty:** different algorithms and distance measures can produce different partitions; internal metrics are not external validation.

### Recommended next steps

1. Evaluate cluster patterns against independent, georeferenced flood-event records if reliable coverage is available.
2. Run spatial and temporal sensitivity tests, including alternative coverage cutoffs, catchment-area criteria, and rainfall periods.
3. Document exact preprocessing parameters and feature-selection rationale.
4. Record package versions and confirm that a clean environment can execute the full pipeline from raw inputs.
5. Report cluster-level overlap, small-cluster behavior, seed stability, and differences between candidate solutions.
6. Treat any future flood-risk interpretation as a separate validation task against observed outcomes.

## References and acknowledgements

1. **India Meteorological Department (IMD).** *Daily Gridded Rainfall Data Set over India, 0.25° × 0.25°, NetCDF.* [Dataset page](https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html). For academic citation, consult the dataset page’s recommended reference: Pai et al. (2014), “Development of a new high spatial resolution (0.25° × 0.25°) long period daily gridded rainfall data set over India and its comparison with existing data sets over the region,” *MAUSAM*, 65(1), 1–18.
2. **HydroBASINS / HydroSHEDS.** *Global sub-basin boundary data.* [Product page and documentation](https://www.hydrosheds.org/products/hydrobasins). Follow applicable terms and attribution requirements.
3. **U.S. Geological Survey (USGS), EROS.** *Shuttle Radar Topography Mission (SRTM) 1 Arc-Second Global.* [Dataset documentation](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm-1/).
4. **scikit-learn.** [Clustering documentation](https://scikit-learn.org/stable/modules/clustering.html) and [PCA documentation](https://scikit-learn.org/stable/modules/decomposition.html#pca).
5. **Streamlit.** [Documentation](https://docs.streamlit.io/).

Please retain dataset attribution and follow each provider’s current terms when publishing code or derived data.

---

## Disclaimer

This repository is an academic research and visualization project. Its outputs must not be used for emergency decisions, public warnings, or site-specific safety decisions. Cluster membership represents similarity in selected rainfall–terrain input features and has not been calibrated to the probability or severity of observed flash floods.
