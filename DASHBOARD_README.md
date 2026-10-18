# Flash-Flood Pattern Atlas dashboard

## Run locally

From the project root folder:

```bash
python -m pip install -r requirements-dashboard.txt
streamlit run app.py
```

The app reads existing outputs from `data/processed/` and the catchment GeoPackage. It does not retrain K-Means. The PCA page computes a two-component projection from the same nine features for visual inspection only.

## Pages

- **Overview** — sample size, selected solution, cluster counts, interpretation guardrails.
- **Catchment map** — HydroBASINS catchments coloured by cluster, where the GeoPackage is available.
- **Cluster profiles** — original-unit profiles and standardised profile heatmap.
- **PCA explorer** — two-dimensional projection and component loadings.
- **Validation & methods** — K-Means sweep, seed stability and any available distance/hierarchical comparison CSVs.
- **Dataset explorer** — feature filters and downloadable catchment assignments.

## Important limitations

The clusters represent exploratory rainfall–terrain similarity. They are not validated flood-risk classes or flood probabilities because independent observed flood outcomes were not used. Keep cluster labels neutral (e.g. Cluster 0) and report overlap, small clusters, data coverage, and model-selection uncertainty.

The distance and hierarchical comparison tables appear only when their output CSVs have been copied into `data/processed/` under these names:

- `final_distance_method_comparison.csv`
- `final_hierarchical_comparison.csv`

The app deliberately does not fabricate missing comparison data.
