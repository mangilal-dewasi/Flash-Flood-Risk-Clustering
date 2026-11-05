"""Build an area-weighted daily rainfall panel for the 113 modelled catchments.

Expected project files:
  data/RF25_2019.nc ... data/RF25_2024.nc
  data/processed/final_catchments_L8_50km2.gpkg
  data/processed/clustering_dataset.csv

Output:
  data/processed/catchment_daily_rainfall_2019_2024.csv

This uses complete IMD grid cells and the same equal-area intersection-weighted
catchment mean as the project's existing extract_rainfall_features.py pipeline.
"""
from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CATCHMENT_FILE = PROCESSED_DIR / "final_catchments_L8_50km2.gpkg"
CLUSTER_DATA_FILE = PROCESSED_DIR / "clustering_dataset.csv"
OUTPUT_FILE = PROCESSED_DIR / "catchment_daily_rainfall_2019_2024.csv"
EQUAL_AREA_CRS = "EPSG:6933"
GRID_RESOLUTION = 0.25


def load_rainfall() -> xr.DataArray:
    paths = sorted(DATA_DIR.glob("RF25_20*.nc"))
    required_years = list(range(2019, 2025))
    year_paths = {}
    for path in paths:
        for year in required_years:
            if path.name == f"RF25_{year}.nc":
                year_paths[year] = path
    missing_years = [year for year in required_years if year not in year_paths]
    if missing_years:
        raise FileNotFoundError(f"Missing annual IMD rainfall files for: {missing_years}")

    catchments = gpd.read_file(CATCHMENT_FILE)
    if catchments.crs is None:
        raise ValueError("Catchment GeoPackage has no CRS.")
    bounds = catchments.to_crs("EPSG:4326").total_bounds
    minx, miny, maxx, maxy = bounds

    annual_arrays = []
    expected_dates = pd.date_range("2019-01-01", "2024-12-31", freq="D")
    for year in required_years:
        with xr.open_dataset(year_paths[year], engine="scipy") as ds:
            if "RAINFALL" not in ds:
                raise ValueError(f"RAINFALL variable not found in {year_paths[year].name}")
            da = ds["RAINFALL"]
            if not np.all(np.diff(da["LATITUDE"].values) > 0):
                raise ValueError("Expected ascending LATITUDE coordinates.")
            if not np.all(np.diff(da["LONGITUDE"].values) > 0):
                raise ValueError("Expected ascending LONGITUDE coordinates.")
            da = da.sel(
                LATITUDE=slice(miny - GRID_RESOLUTION, maxy + GRID_RESOLUTION),
                LONGITUDE=slice(minx - GRID_RESOLUTION, maxx + GRID_RESOLUTION),
            ).load()
            annual_arrays.append(da)

    rainfall = xr.concat(annual_arrays, dim="TIME").sortby("TIME")
    actual_dates = pd.DatetimeIndex(rainfall["TIME"].values)
    if not actual_dates.equals(expected_dates):
        raise ValueError("Annual files do not form a continuous 2019-2024 daily series.")
    return rainfall


def main() -> None:
    for path in [CATCHMENT_FILE, CLUSTER_DATA_FILE]:
        if not path.exists():
            raise FileNotFoundError(path)

    print("Loading IMD rainfall grid for 2019-2024...")
    rainfall = load_rainfall()
    catchments = gpd.read_file(CATCHMENT_FILE)
    catchments["catchment_id"] = catchments["HYBAS_ID"].astype(str)
    if catchments.crs.to_epsg() != 4326:
        catchments = catchments.to_crs(4326)

    model_ids = set(
        pd.read_csv(CLUSTER_DATA_FILE, dtype={"catchment_id": str})["catchment_id"]
    )
    catchments = catchments[catchments["catchment_id"].isin(model_ids)].copy()
    if catchments["catchment_id"].nunique() != 113:
        raise ValueError(
            "Expected 113 catchments to match clustering_dataset.csv; "
            f"found {catchments['catchment_id'].nunique()}."
        )

    valid_mask = rainfall.notnull().all(dim="TIME").values
    latitudes = rainfall["LATITUDE"].values
    longitudes = rainfall["LONGITUDE"].values
    half = GRID_RESOLUTION / 2
    cells = []
    for i, lat in enumerate(latitudes):
        for j, lon in enumerate(longitudes):
            if valid_mask[i, j]:
                cells.append(
                    {
                        "i": i,
                        "j": j,
                        "lat": float(lat),
                        "lon": float(lon),
                        "geometry": box(lon - half, lat - half, lon + half, lat + half),
                    }
                )
    grid = gpd.GeoDataFrame(cells, crs="EPSG:4326")

    grid_equal = grid.to_crs(EQUAL_AREA_CRS)
    catchments_equal = catchments[["catchment_id", "geometry"]].to_crs(EQUAL_AREA_CRS)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="keep_geom_type=True in overlay")
        intersections = gpd.overlay(grid_equal, catchments_equal, how="intersection")
    intersections["intersection_area_m2"] = intersections.geometry.area
    intersections = intersections[intersections["intersection_area_m2"] > 0].copy()
    catchment_area = catchments_equal.set_index("catchment_id").geometry.area

    time_index = pd.DatetimeIndex(rainfall["TIME"].values)
    rainfall_values = rainfall.values.astype("float32")
    records = []
    coverage_rows = []
    for catchment_id in sorted(model_ids):
        local = intersections[intersections["catchment_id"] == catchment_id].sort_values(["i", "j"])
        if local.empty:
            raise ValueError(f"No complete rainfall grid cells intersect catchment {catchment_id}.")
        ii = local["i"].to_numpy(dtype=int)
        jj = local["j"].to_numpy(dtype=int)
        weights = local["intersection_area_m2"].to_numpy(dtype="float64")
        spatial_coverage = float(weights.sum() / catchment_area.loc[catchment_id])
        if spatial_coverage < 0.90:
            raise ValueError(
                f"Catchment {catchment_id} has only {spatial_coverage:.1%} rainfall spatial coverage."
            )
        cell_values = rainfall_values[:, ii, jj]
        daily_values = np.average(cell_values, axis=1, weights=weights).astype("float32")
        records.append(
            pd.DataFrame(
                {
                    "catchment_id": catchment_id,
                    "date": time_index,
                    "rainfall_mm": daily_values,
                }
            )
        )
        coverage_rows.append(
            {
                "catchment_id": catchment_id,
                "rainfall_grid_cells": len(local),
                "spatial_coverage_fraction": spatial_coverage,
            }
        )

    panel = pd.concat(records, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["catchment_id", "date"]).reset_index(drop=True)
    panel["month"] = panel["date"].dt.month.astype("int8")
    panel["day_of_year"] = panel["date"].dt.dayofyear.astype("int16")
    panel["is_monsoon"] = panel["month"].isin([6, 7, 8, 9]).astype("int8")

    grouped = panel.groupby("catchment_id", sort=False)["rainfall_mm"]
    for lag in [1, 2, 3, 7]:
        panel[f"rain_lag_{lag}d_mm"] = grouped.shift(lag)
    for window in [3, 7]:
        panel[f"rain_sum_prev_{window}d_mm"] = grouped.transform(
            lambda series: series.shift(1).rolling(window, min_periods=window).sum()
        )
        panel[f"rain_max_prev_{window}d_mm"] = grouped.transform(
            lambda series: series.shift(1).rolling(window, min_periods=window).max()
        )
    panel["target_next_day_rainfall_mm"] = grouped.shift(-1)
    panel["target_next_day_ge_50mm"] = (
        panel["target_next_day_rainfall_mm"] >= 50
    ).astype("float32")
    panel["target_next_day_ge_100mm"] = (
        panel["target_next_day_rainfall_mm"] >= 100
    ).astype("float32")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUTPUT_FILE, index=False, float_format="%.5f")
    coverage = pd.DataFrame(coverage_rows)
    print(f"Saved {len(panel):,} rows to {OUTPUT_FILE}")
    print(
        "Spatial coverage fraction: "
        f"min={coverage.spatial_coverage_fraction.min():.3f}, "
        f"mean={coverage.spatial_coverage_fraction.mean():.3f}"
    )
    print("Source period: 2019-01-01 to 2024-12-31 (2,192 daily records per catchment)")


if __name__ == "__main__":
    main()
