# 🌧️ Flash-Flood Pattern Atlas

### Exploratory clustering of rainfall–terrain patterns across Himalayan catchments

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/) [![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/) [![Method](https://img.shields.io/badge/Method-Unsupervised%20Learning-0F766E)](#methodology)

**Repository:** [mangilal-dewasi/Flash-Flood-Risk-Clustering](https://github.com/mangilal-dewasi/Flash-Flood-Risk-Clustering)

**INT396 · Unsupervised Learning Research Project**  
**Study region:** Himachal Pradesh and Uttarakhand, India  
**Study period:** 2019–2024

The **Flash-Flood Pattern Atlas** explores how hilly catchments differ in their monsoon rainfall characteristics and terrain. It combines daily gridded rainfall, terrain metrics derived from elevation data, and HydroBASINS catchment boundaries. Unsupervised-learning methods are then used to group catchments with similar feature profiles and explore how those groups vary spatially.

> **Important research boundary:** The clusters are exploratory rainfall–terrain pattern groups. They are **not validated flood-risk classes, flood forecasts, or estimates of flood probability**. Independent observed flood-event labels were not used to train or validate this analysis.

---

## Contents

- [Project at a glance](#project-at-a-glance)
- [Research questions](#research-questions)
- [Data sources](#data-sources)
- [Features used](#features-used)
- [Methodology](#methodology)
- [Preliminary findings](#preliminary-findings)
- [Interactive dashboard](#interactive-dashboard)
- [Repository structure](#repository-structure)
- [Installation and use](#installation-and-use)
- [Reproducibility notes](#reproducibility-notes)
- [Limitations and next steps](#limitations-and-next-steps)
- [References and acknowledgements](#references-and-acknowledgements)

## Project at a glance

| Item | Description |
|---|---|
| Objective | Discover groups of catchments with similar rainfall and terrain characteristics |
| Study area | Hilly catchments in Himachal Pradesh and Uttarakhand, India |
| Time window | Daily rainfall data from 2019 through 2024 |
| Catchment framework | HydroBASINS Level 8 catchments; study workflow applies a 50 km² minimum-area criterion |
| Modelling dataset | 113 catchments, 9 selected numeric features, no missing/non-finite feature values in the final modelling table |
| Main algorithm | K-Means clustering after feature standardisation |
| Additional analysis | Cluster validation, random-seed stability, hierarchical clustering, PCA, and a Manhattan-distance medoid-style comparison |
| Dashboard | Streamlit, with catchment mapping, cluster profiles, PCA exploration, validation metrics, and a dataset explorer |

## Research questions

1. Do catchments in the study area form distinguishable groups based on rainfall and terrain characteristics?
2. How do candidate cluster solutions compare under internal validation metrics and repeated random initialisations?
3. Which features most clearly describe differences between the resulting groups?
4. How can these patterns be explored on a catchment map without treating cluster membership as a measured flood outcome?

## Data sources

The analysis combines three core data types:

| Data | Role in the project | Source |
|---|---|---|
| IMD daily gridded rainfall at 0.25° resolution | Rainfall history and heavy-rainfall indicators for 2019–2024 | [India Meteorological Department — 0.25° daily rainfall NetCDF](https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html) |
| SRTM elevation data | Elevation, relief, slope, and slope variability derived for each catchment | [USGS — SRTM 1 Arc-Second Global](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm-1/) |
| HydroBASINS Level 8 basin polygons | Catchment boundaries and catchment identifiers | [HydroBASINS — HydroSHEDS](https://www.hydrosheds.org/products/hydrobasins) |

Study-boundary files, including the state/NWIC boundary layer where used, support geographic selection of the study region. They are not among the nine clustering features.

### Data handling

Large source datasets—especially the annual NetCDF rainfall files and SRTM tiles—should normally remain outside ordinary Git commits. The repository should retain the processed tables needed by the dashboard, the analysis scripts, source links, and clear download/preparation instructions. Always follow the data provider's terms and attribution requirements before redistributing source files.

## Features used

The modelling dataset contains the following nine features:

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

Rainfall features are derived from the 2019–2024 daily record, with monsoon months defined in the extraction script as June–September. Heavy and very-heavy daily rainfall thresholds are 50 mm and 100 mm, respectively.

## Methodology

1. **Prepare the study area.** Select HydroBASINS Level 8 catchments intersecting the study region and apply the study workflow's catchment-area criterion.
2. **Extract rainfall features.** Use the IMD 0.25° gridded daily rainfall record to derive catchment-level rainfall measures, using area-weighted aggregation across intersecting grid cells.
3. **Derive terrain features.** Use SRTM elevation tiles to calculate catchment elevation and slope summaries.
4. **Build the modelling table.** Join rainfall and terrain features by catchment identifier, inspect coverage, and retain valid rows.
5. **Select and standardise features.** Use the nine documented variables and standardise them before distance-based clustering so differences in measurement units do not dominate by scale alone.
6. **Evaluate K-Means solutions.** Test cluster counts from *k = 2* to *k = 8*, using internal metrics such as silhouette score, Davies–Bouldin index, and Calinski–Harabasz score.
7. **Check robustness and compare approaches.** Re-run candidate solutions across multiple random seeds; compare cluster stability and conduct supplementary hierarchical, PCA, and Manhattan-distance medoid-style analyses.
8. **Explore the results.** Use the Streamlit dashboard to inspect cluster assignments, feature profiles, catchment patterns, and validation summaries.

PCA is used as a dimensionality-reduction and visualisation aid. The primary K-Means model is fit using all nine standardised features, not just the two plotted PCA components.

## Preliminary findings

The current consolidated analysis produced the following results on 113 catchments:

| Diagnostic | K = 5 | K = 6 |
|---|---:|---:|
| Overall silhouette score | **0.3257** | 0.3235 |
| Davies–Bouldin index | **1.0270** | 1.0275 |
| Mean pairwise adjusted Rand index across 10 seeds | 0.9110 | **0.9913** |
| Minimum pairwise adjusted Rand index across 10 seeds | 0.7492 | **0.9809** |
| Smallest cluster in the reported solution | 9 catchments | 7 catchments |

The current analysis therefore retains both solutions for interpretation: **k = 5 has marginally stronger internal separation metrics, while k = 6 shows substantially higher stability across tested random seeds**. Neither result alone establishes that the groups correspond to actual flood events or real-world risk levels.

The first two principal components explain approximately **73.72%** of the standardised feature variance (PC1: 50.14%; PC2: 23.58%). This is useful for visual exploration, but a two-dimensional projection cannot preserve every relationship in the full feature space.

These are exploratory results and may change if the input data, feature choices, preprocessing, or analysis code changes. Refer to the generated report and CSV outputs for the detailed metrics and cluster profiles.

## Interactive dashboard

The Streamlit app provides these sections:

- **Overview** — project snapshot, cluster counts, and interpretation guardrails.
- **Catchment map** — catchments coloured by exploratory cluster membership, when the GeoPackage is available.
- **Cluster profiles** — feature summaries in original units and standardised profiles.
- **PCA explorer** — two-component projection and component loadings.
- **Validation & methods** — K-Means evaluation, seed stability, and available supplementary comparisons.
- **Dataset explorer** — filter and download the processed catchment-level data and assignments.

### Run the dashboard

Run the following commands from the repository root, where `app.py` and `requirements-dashboard.txt` are located:

```bash
python -m venv .venv
```

Activate the environment, then install the dashboard dependencies:

**Windows PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dashboard.txt
streamlit run app.py
```

**macOS / Linux**

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dashboard.txt
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

The dashboard reads existing result files from `data/processed/`; it does **not** retrain K-Means when it starts. Ensure the processed CSVs are in place. The catchment map also requires `data/processed/final_catchments_L8_50km2.gpkg`. The app handles optional comparison files gracefully when they are not present.

## Repository structure

The repository is organised around source code, processed outputs, and analysis reports. A typical layout is:

```text
Flash-Flood-Risk-Clustering/
├── app.py
├── requirements-dashboard.txt
├── create_final_catchments.py
├── get_srtm_tiles.py
├── data/
│   ├── hydrobasins/              # downloaded catchment source data
│   ├── interim/                  # study boundaries and prepared inputs
│   ├── raw/srtm/                 # local SRTM elevation tiles
│   └── processed/                # modelling tables, assignments, profiles
├── src/
│   ├── extract_rainfall_features.py
│   ├── extract_terrain_features.py
│   ├── build_modeling_dataset.py
│   ├── select_clustering_features.py
│   ├── run_kmeans_experiments.py
│   ├── analyze_kmeans_clusters.py
│   ├── analyze_six_clusters.py
│   ├── test_clusters_stability.py
│   ├── final_project_analysis.py
│   └── final_project_analysis_pca_manhattan.py
└── reports/
    ├── final_analysis_report.txt
    └── figures/
```

Some source datasets and generated files may be omitted from a GitHub clone because of size. Check the required paths at the top of each preprocessing script before running the complete pipeline; a script may expect an intermediate file to have been prepared locally.

## Reproducibility notes

- Run commands from the repository root so relative paths resolve correctly.
- Use the exact rainfall period (2019–2024), monsoon months (June–September), and rainfall thresholds documented above when recreating the feature table.
- The analysis code uses `StandardScaler` before K-Means. Keep preprocessing and scaling consistent when comparing experiments.
- Random seeds and the number of initialisations affect clustering. Record these settings when modifying scripts.
- Preserve catchment identifiers as strings when reading or writing CSVs to avoid accidental numeric conversion.
- The analysis scripts produce overlapping `final_*` output names. Before running multiple alternative final-analysis scripts, preserve any results you need to compare and note which script generated each output.
- Do not compare raw objective values between algorithms that optimise different objective functions as if they were directly equivalent.

## Limitations and next steps

### Current limitations

- **No independent observed flood labels:** This is pattern discovery, not supervised flood prediction or validated risk classification.
- **Moderate cluster separation:** Internal validation metrics indicate some separation, but the groups are not perfectly distinct.
- **Small clusters:** Some clusters contain relatively few catchments and should be interpreted cautiously.
- **Feature dependence:** Several rainfall variables are correlated, so similar information may contribute through more than one feature.
- **Spatial and temporal scope:** Findings are tied to the selected catchments and the 2019–2024 rainfall period; they should not be generalised automatically to other regions or periods.
- **Input-data uncertainty:** Gridded rainfall and terrain products have resolution and measurement limitations that may affect catchment summaries.

### Recommended next steps

1. Validate the discovered patterns against independent, georeferenced flood-event, flash-flood, or landslide records where coverage and quality permit.
2. Conduct spatial and temporal sensitivity checks, including alternative catchment filters and rainfall periods.
3. Document the source, version, preprocessing history, and licence/terms for every input dataset.
4. Add a complete environment specification for the full preprocessing workflow, separate from dashboard-only dependencies.
5. Treat any future risk interpretation as a separate validation task, with transparent evaluation against observed outcomes.

## References and acknowledgements

1. **India Meteorological Department (IMD).** *Daily Gridded Rainfall Data Set over India, 0.25° × 0.25°, NetCDF.* [Dataset page](https://www.imdpune.gov.in/cmpg/Griddata/Rainfall_25_NetCDF.html). For academic use of this rainfall product, consult the IMD page's recommended citation: Pai et al. (2014), *Development of a new high spatial resolution (0.25° × 0.25°) long period daily gridded rainfall data set over India and its comparison with existing data sets over the region*, **MAUSAM**, 65(1), 1–18.
2. **HydroBASINS / HydroSHEDS.** *Global sub-basin boundary data.* [Product page and documentation](https://www.hydrosheds.org/products/hydrobasins). See the product page for the applicable terms and attribution requirements.
3. **U.S. Geological Survey (USGS), EROS.** *Shuttle Radar Topography Mission (SRTM) 1 Arc-Second Global.* [Dataset documentation](https://www.usgs.gov/centers/eros/science/usgs-eros-archive-digital-elevation-shuttle-radar-topography-mission-srtm-1/).
4. **scikit-learn.** [Clustering documentation](https://scikit-learn.org/stable/modules/clustering.html), [PCA documentation](https://scikit-learn.org/stable/modules/decomposition.html#pca).
5. **Streamlit.** [Documentation](https://docs.streamlit.io/).

Please retain the original dataset attribution and follow each provider's current terms when publishing derived data or code.

---

## Disclaimer

This repository is an academic research and visualisation project. Its outputs should not be used for emergency decisions, public warning, or site-specific safety decisions. Cluster membership reflects similarity in the selected input features and has not been calibrated to the probability or severity of observed flash floods.
