from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

from shapely.geometry import box


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "imd_rainfall_2019_2024_combined.nc"
)

CATCHMENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_catchments_L8_50km2.gpkg"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "catchment_rainfall_features.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

MONSOON_MONTHS = [6, 7, 8, 9]

HEAVY_RAIN_THRESHOLD = 50.0
VERY_HEAVY_RAIN_THRESHOLD = 100.0

GRID_RESOLUTION = 0.25

EQUAL_AREA_CRS = "EPSG:6933"


# ============================================================
# RAINFALL FEATURES
# ============================================================

def calculate_rainfall_features(rainfall, weights):
    """
    Calculate rainfall features from an area-weighted
    catchment rainfall time series.

    rainfall:
        TIME x CELL

    weights:
        intersection area for each rainfall cell.
    """

    # --------------------------------------------------------
    # Area-weighted daily catchment rainfall
    # --------------------------------------------------------

    weight_da = xr.DataArray(
        weights,
        dims="cell",
    )

    daily = (
        rainfall
        .weighted(weight_da)
        .mean(dim="cell")
    )

    daily_series = (
        daily
        .to_series()
        .sort_index()
        .dropna()
    )

    if daily_series.empty:
        return None

    # --------------------------------------------------------
    # Temporal coverage
    # --------------------------------------------------------

    total_days = len(
        rainfall["TIME"]
    )

    valid_days = int(
        len(daily_series)
    )

    temporal_coverage = (
        valid_days / total_days
    )

    # --------------------------------------------------------
    # Annual rainfall
    # --------------------------------------------------------

    annual_total = (
        daily_series
        .resample("YS")
        .sum()
    )

    mean_annual_rainfall = float(
        annual_total.mean()
    )

    # --------------------------------------------------------
    # Monsoon rainfall
    # --------------------------------------------------------

    monsoon = daily_series[
        daily_series.index.month.isin(
            MONSOON_MONTHS
        )
    ]

    monsoon_annual_total = (
        monsoon
        .resample("YS")
        .sum()
    )

    mean_monsoon_rainfall = float(
        monsoon_annual_total.mean()
    )

    # --------------------------------------------------------
    # Maximum daily rainfall
    # --------------------------------------------------------

    annual_max_daily = (
        daily_series
        .resample("YS")
        .max()
    )

    mean_annual_max_daily = float(
        annual_max_daily.mean()
    )

    # --------------------------------------------------------
    # Heavy rainfall days >= 50 mm
    # --------------------------------------------------------

    heavy_days = (
        daily_series
        >= HEAVY_RAIN_THRESHOLD
    )

    annual_heavy_days = (
        heavy_days
        .resample("YS")
        .sum()
    )

    mean_heavy_days = float(
        annual_heavy_days.mean()
    )

    # --------------------------------------------------------
    # Very heavy rainfall days >= 100 mm
    # --------------------------------------------------------

    very_heavy_days = (
        daily_series
        >= VERY_HEAVY_RAIN_THRESHOLD
    )

    annual_very_heavy_days = (
        very_heavy_days
        .resample("YS")
        .sum()
    )

    mean_very_heavy_days = float(
        annual_very_heavy_days.mean()
    )

    # --------------------------------------------------------
    # Maximum 3-day rainfall
    # --------------------------------------------------------

    rolling_3day = (
        daily_series
        .rolling(
            window=3,
            min_periods=3,
        )
        .sum()
    )

    annual_max_3day = (
        rolling_3day
        .resample("YS")
        .max()
    )

    mean_max_3day = float(
        annual_max_3day.mean()
    )

    # --------------------------------------------------------
    # Inter-annual rainfall variability
    # --------------------------------------------------------

    annual_mean = float(
        annual_total.mean()
    )

    annual_std = float(
        annual_total.std()
    )

    if annual_mean > 0:
        rainfall_cv = (
            annual_std / annual_mean
        )
    else:
        rainfall_cv = np.nan

    return {
        "mean_annual_rainfall_mm":
            mean_annual_rainfall,

        "mean_monsoon_rainfall_mm":
            mean_monsoon_rainfall,

        "mean_annual_max_daily_mm":
            mean_annual_max_daily,

        "mean_heavy_rain_days_50mm":
            mean_heavy_days,

        "mean_very_heavy_rain_days_100mm":
            mean_very_heavy_days,

        "mean_max_3day_rainfall_mm":
            mean_max_3day,

        "rainfall_cv":
            rainfall_cv,

        "rainfall_valid_days":
            valid_days,

        "rainfall_temporal_coverage_fraction":
            temporal_coverage,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("FLASH-FLOOD PATTERN ATLAS")
    print("AREA-WEIGHTED IMD RAINFALL FEATURE EXTRACTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load rainfall
    # --------------------------------------------------------

    print("\nLoading IMD rainfall...")

    ds = xr.open_dataset(
        RAIN_FILE
    )

    rainfall = ds["RAINFALL"]

    latitudes = rainfall[
        "LATITUDE"
    ].values

    longitudes = rainfall[
        "LONGITUDE"
    ].values

    print(
        "Dimensions:",
        rainfall.sizes
    )

    # --------------------------------------------------------
    # Identify complete rainfall cells
    # --------------------------------------------------------

    print(
        "\nChecking rainfall-cell completeness..."
    )

    valid_cell_mask = (
        rainfall
        .notnull()
        .all(dim="TIME")
        .values
    )

    valid_cell_count = int(
        valid_cell_mask.sum()
    )

    total_cell_count = (
        valid_cell_mask.size
    )

    print(
        f"Complete cells: "
        f"{valid_cell_count}/{total_cell_count}"
    )

    # --------------------------------------------------------
    # Build polygons for complete IMD cells
    # --------------------------------------------------------

    grid_records = []

    half = GRID_RESOLUTION / 2

    for i, lat in enumerate(latitudes):

        for j, lon in enumerate(longitudes):

            if not valid_cell_mask[i, j]:
                continue

            polygon = box(
                float(lon) - half,
                float(lat) - half,
                float(lon) + half,
                float(lat) + half,
            )

            grid_records.append(
                {
                    "lat": float(lat),
                    "lon": float(lon),
                    "geometry": polygon,
                }
            )

    grid = gpd.GeoDataFrame(
        grid_records,
        crs="EPSG:4326",
    )

    print(
        "Rainfall polygons:",
        len(grid)
    )

    # --------------------------------------------------------
    # Load catchments
    # --------------------------------------------------------

    print(
        "\nLoading catchments..."
    )

    catchments = gpd.read_file(
        CATCHMENT_FILE
    )

    if catchments.crs.to_epsg() != 4326:
        catchments = catchments.to_crs(
            4326
        )

    catchments = catchments.copy()

    catchments["catchment_id"] = (
        catchments["HYBAS_ID"]
        .astype(str)
    )

    print(
        "Catchments:",
        len(catchments)
    )

    # --------------------------------------------------------
    # Equal-area projection
    # --------------------------------------------------------

    print(
        "\nProjecting to equal-area CRS..."
    )

    grid_equal = grid.to_crs(
        EQUAL_AREA_CRS
    )

    catchments_equal = (
        catchments[
            [
                "catchment_id",
                "geometry",
            ]
        ]
        .to_crs(EQUAL_AREA_CRS)
    )

    # --------------------------------------------------------
    # Calculate grid/catchment intersections
    # --------------------------------------------------------

    print(
        "\nCalculating grid-cell/catchment intersections..."
    )

    intersections = gpd.overlay(
        grid_equal,
        catchments_equal,
        how="intersection",
    )

    intersections[
        "intersection_area_m2"
    ] = intersections.geometry.area

    intersections = intersections[
        intersections[
            "intersection_area_m2"
        ] > 0
    ].copy()

    print(
        "Intersection records:",
        len(intersections)
    )

    # --------------------------------------------------------
    # Catchment total areas
    # --------------------------------------------------------

    catchment_area = (
        catchments_equal
        .set_index("catchment_id")
        .geometry
        .area
    )

    # --------------------------------------------------------
    # Process catchments
    # --------------------------------------------------------

    records = []

    print(
        "\nExtracting rainfall features..."
    )

    for index, catchment in catchments.iterrows():

        catchment_id = str(
            catchment["HYBAS_ID"]
        )

        print(
            f"\n[{index + 1:3d}/"
            f"{len(catchments)}] "
            f"Catchment {catchment_id}"
        )

        local = intersections[
            intersections[
                "catchment_id"
            ] == catchment_id
        ].copy()

        if local.empty:

            print(
                "  No complete IMD rainfall "
                "cells intersect catchment."
            )

            records.append(
                {
                    "catchment_id":
                        catchment_id,

                    "rainfall_grid_cells":
                        0,

                    "rainfall_spatial_coverage_fraction":
                        0.0,

                    "mean_annual_rainfall_mm":
                        np.nan,

                    "mean_monsoon_rainfall_mm":
                        np.nan,

                    "mean_annual_max_daily_mm":
                        np.nan,

                    "mean_heavy_rain_days_50mm":
                        np.nan,

                    "mean_very_heavy_rain_days_100mm":
                        np.nan,

                    "mean_max_3day_rainfall_mm":
                        np.nan,

                    "rainfall_cv":
                        np.nan,

                    "rainfall_valid_days":
                        0,

                    "rainfall_temporal_coverage_fraction":
                        0.0,
                }
            )

            continue

        # ----------------------------------------------------
        # Spatial weights
        # ----------------------------------------------------

        local = local.sort_values(
            ["lat", "lon"]
        )

        cell_lats = (
            local["lat"]
            .astype(float)
            .tolist()
        )

        cell_lons = (
            local["lon"]
            .astype(float)
            .tolist()
        )

        weights = (
            local[
                "intersection_area_m2"
            ]
            .astype(float)
            .values
        )

        # ----------------------------------------------------
        # Spatial coverage
        # ----------------------------------------------------

        valid_area = float(
            weights.sum()
        )

        total_area = float(
            catchment_area[
                catchment_id
            ]
        )

        spatial_coverage = (
            valid_area / total_area
        )

        spatial_coverage = min(
            1.0,
            max(
                0.0,
                spatial_coverage,
            ),
        )

        print(
            f"  Complete IMD cells: "
            f"{len(local)}"
        )

        print(
            f"  Valid rainfall area: "
            f"{spatial_coverage * 100:.2f}%"
        )

        # ----------------------------------------------------
        # Extract rainfall cells
        # ----------------------------------------------------

        catchment_rainfall = (
            rainfall.sel(
                LATITUDE=xr.DataArray(
                    cell_lats,
                    dims="cell",
                ),
                LONGITUDE=xr.DataArray(
                    cell_lons,
                    dims="cell",
                ),
            )
        )

        # ----------------------------------------------------
        # Calculate features
        # ----------------------------------------------------

        features = (
            calculate_rainfall_features(
                catchment_rainfall,
                weights,
            )
        )

        if features is None:

            print(
                "  WARNING: empty rainfall series."
            )

            features = {
                "mean_annual_rainfall_mm":
                    np.nan,

                "mean_monsoon_rainfall_mm":
                    np.nan,

                "mean_annual_max_daily_mm":
                    np.nan,

                "mean_heavy_rain_days_50mm":
                    np.nan,

                "mean_very_heavy_rain_days_100mm":
                    np.nan,

                "mean_max_3day_rainfall_mm":
                    np.nan,

                "rainfall_cv":
                    np.nan,

                "rainfall_valid_days":
                    0,

                "rainfall_temporal_coverage_fraction":
                    0.0,
            }

        record = {
            "catchment_id":
                catchment_id,

            "rainfall_grid_cells":
                len(local),

            "rainfall_spatial_coverage_fraction":
                spatial_coverage,
        }

        record.update(
            features
        )

        records.append(record)

        print(
            f"  Annual rainfall: "
            f"{features['mean_annual_rainfall_mm']:.2f} mm"
        )

        print(
            f"  Monsoon rainfall: "
            f"{features['mean_monsoon_rainfall_mm']:.2f} mm"
        )

        print(
            f"  Max daily rainfall: "
            f"{features['mean_annual_max_daily_mm']:.2f} mm"
        )

        print(
            f"  Max 3-day rainfall: "
            f"{features['mean_max_3day_rainfall_mm']:.2f} mm"
        )

    # --------------------------------------------------------
    # Create dataframe
    # --------------------------------------------------------

    df = pd.DataFrame(
        records
    )

    # --------------------------------------------------------
    # Quality control
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "AREA-WEIGHTED RAINFALL EXTRACTION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nRows:",
        len(df)
    )

    print(
        "Columns:",
        len(df.columns)
    )

    print(
        "\nMissing values:"
    )

    print(
        df.isna().sum()
    )

    print(
        "\nRainfall grid-cell coverage:"
    )

    print(
        df[
            "rainfall_grid_cells"
        ].describe()
    )

    print(
        "\nSpatial coverage:"
    )

    print(
        df[
            "rainfall_spatial_coverage_fraction"
        ]
        .describe()
    )

    print(
        "\nTemporal coverage:"
    )

    print(
        df[
            "rainfall_temporal_coverage_fraction"
        ]
        .describe()
    )

    feature_columns = [
        "mean_annual_rainfall_mm",
        "mean_monsoon_rainfall_mm",
        "mean_annual_max_daily_mm",
        "mean_heavy_rain_days_50mm",
        "mean_very_heavy_rain_days_100mm",
        "mean_max_3day_rainfall_mm",
        "rainfall_cv",
    ]

    print(
        "\nRainfall feature summary:"
    )

    print(
        df[
            feature_columns
        ]
        .describe()
        .T
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\nSaved:",
        OUTPUT_FILE
    )

    ds.close()


if __name__ == "__main__":
    main()