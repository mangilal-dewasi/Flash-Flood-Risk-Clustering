# Catchment Pattern Explorer — Rainfall Prediction Add-on

This add-on adds a **historical one-day-ahead catchment-average rainfall estimate** to the existing Catchment Pattern Explorer. It does not add flood prediction, flood probability, or an early-warning alert.

## Files in this add-on

- `catchment_pattern_explorer.py` — updated page module with three tabs: pattern matching, catchment comparison, and 24-hour rainfall estimate.
- `data/processed/catchment_daily_rainfall_2019_2024.csv` — area-weighted daily rainfall panel for the 113 modelled catchments, 2019–2024.
- `models/next_day_rainfall_lgbm.joblib` — trained LightGBM Tweedie regression model.
- `src/build_catchment_daily_timeseries.py` — reproducible daily catchment rainfall extraction script from annual `RF25_2019.nc` ... `RF25_2024.nc` files.
- `src/train_next_day_rainfall_model.py` — reproducible model training script.
- `reports/rainfall_forecast_validation.md` — task definition, chronological validation, metrics and limitations.

## Install into the existing project

Unzip this add-on at the **project root** (the same folder that contains `streamlit_app.py`, `data/`, and `src/`). Allow it to replace the existing `catchment_pattern_explorer.py`.

The following project-relative files should exist afterward:

```text
INT396 Project/
├── streamlit_app.py                    # keep your existing file unchanged
├── catchment_pattern_explorer.py       # replace with this updated module
├── models/
│   └── next_day_rainfall_lgbm.joblib
├── data/
│   └── processed/
│       ├── clustering_dataset.csv      # existing project file; keep it
│       └── catchment_daily_rainfall_2019_2024.csv
└── src/
    ├── build_catchment_daily_timeseries.py
    └── train_next_day_rainfall_model.py
```

Add this dependency to the requirements file used by your local environment and Streamlit deployment:

```text
lightgbm>=4.0
```

The project already uses scikit-learn, pandas, NumPy and GeoPandas. The daily panel is bundled, so you do not need xarray merely to run the dashboard. To rebuild the panel from the annual NetCDF files, make sure `xarray` is also installed.

## Keep your existing `streamlit_app.py`

No full-app replacement is needed. Keep the import and page branch you already added:

```python
from catchment_pattern_explorer import render_pattern_explorer
```

```python
elif page == "Catchment Pattern Explorer":
    render_pattern_explorer(DATA)
```

The function name and its required `data_dir` argument remain compatible with this call.

## How the prediction tab works

- It operates on the **113 existing reference catchments**, because an arbitrary new catchment has no daily rainfall history in this dataset.
- It uses an as-of date in the 2024 monsoon. Rainfall observed up to that date is used to estimate the following day's catchment-area mean rainfall.
- The tab shows the model estimate, the actual next-day value from the held-out 2024 record, the absolute error, and the preceding week's rainfall.
- The selected model was trained using 2019–2023 data. The 2024 monsoon is kept as a historical test window.
- This is not a live forecast. The bundled historical rainfall files end on 31 December 2024. A live forecast in 2026 requires updated daily rainfall observations through the current date and a fresh time-aware validation.

## Rebuild the data and model (optional)

From the project root, after dependencies and source files are present:

```bash
python src/build_catchment_daily_timeseries.py
python src/train_next_day_rainfall_model.py
```

The first script expects the six annual IMD files in `data/` and `final_catchments_L8_50km2.gpkg` plus `clustering_dataset.csv` in `data/processed/`. The second script expects the daily panel and the existing clustering dataset.
