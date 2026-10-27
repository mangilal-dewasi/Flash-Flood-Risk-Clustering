
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------
# FLASH-FLOOD PATTERN ATLAS
# Build and validate the final modeling dataset
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

RAINFALL_FILE = DATA_DIR / "catchment_rainfall_features.csv"
TERRAIN_FILE = DATA_DIR / "catchment_terrain_features.csv"

COVERAGE_THRESHOLD = 0.95

RAIN_FEATURES = [
    "mean_annual_rainfall_mm",
    "mean_monsoon_rainfall_mm",
    "mean_annual_max_daily_mm",
    "mean_heavy_rain_days_50mm",
    "mean_very_heavy_rain_days_100mm",
    "mean_max_3day_rainfall_mm",
    "rainfall_cv",
]

TERRAIN_FEATURES = [
    "mean_elevation_m",
    "relief_m",
    "mean_slope_deg",
    "slope_std_deg",
]

FEATURES = RAIN_FEATURES + TERRAIN_FEATURES


def normalize_id(series):
    """Normalize catchment IDs before joining CSV files."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )


def main():
    print("=" * 65)
    print("FLASH-FLOOD PATTERN ATLAS")
    print("BUILDING FINAL MODELING DATASET")
    print("=" * 65)

    # 1. Load the feature tables
    rainfall = pd.read_csv(RAINFALL_FILE)
    terrain = pd.read_csv(TERRAIN_FILE)

    for name, df in [
        ("Rainfall", rainfall),
        ("Terrain", terrain),
    ]:
        if "catchment_id" not in df.columns:
            raise ValueError(f"{name} file has no catchment_id column.")

        if df["catchment_id"].duplicated().any():
            raise ValueError(
                f"{name} table contains duplicate catchment IDs."
            )

        df["catchment_id"] = normalize_id(df["catchment_id"])

    print(f"\nRainfall records: {len(rainfall)}")
    print(f"Terrain records:  {len(terrain)}")

    # 2. Verify required columns
    required_rainfall = RAIN_FEATURES + [
        "rainfall_spatial_coverage_fraction",
        "rainfall_temporal_coverage_fraction",
    ]

    missing_rain = set(required_rainfall) - set(rainfall.columns)
    missing_terrain = set(TERRAIN_FEATURES) - set(terrain.columns)

    if missing_rain:
        raise ValueError(f"Missing rainfall columns: {sorted(missing_rain)}")

    if missing_terrain:
        raise ValueError(f"Missing terrain columns: {sorted(missing_terrain)}")

    # 3. Merge one row per catchment
    merged = rainfall.merge(
        terrain[["catchment_id"] + TERRAIN_FEATURES],
        on="catchment_id",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )

    print("\nMerge results:")
    print(merged["_merge"].value_counts().to_string())

    unmatched = merged.loc[
        merged["_merge"] != "both", ["catchment_id", "_merge"]
    ]

    if not unmatched.empty:
        unmatched.to_csv(
            DATA_DIR / "unmatched_catchments.csv", index=False
        )
        raise ValueError(
            "Rainfall and terrain IDs do not match completely. "
            "See unmatched_catchments.csv before continuing."
        )

    merged = merged.drop(columns="_merge")

    # 4. Save the merged dataset before coverage filtering
    merged.to_csv(
        DATA_DIR / "catchment_features_merged_all.csv",
        index=False,
    )

    # 5. Apply the predefined spatial coverage threshold
    coverage = pd.to_numeric(
        merged["rainfall_spatial_coverage_fraction"],
        errors="coerce",
    )

    eligible_mask = coverage >= COVERAGE_THRESHOLD

    excluded = merged.loc[~eligible_mask].copy()
    eligible = merged.loc[eligible_mask].copy()

    excluded.to_csv(
        DATA_DIR / "rainfall_coverage_exclusions.csv",
        index=False,
    )

    # 6. Convert selected features to numeric values
    for feature in FEATURES:
        eligible[feature] = pd.to_numeric(
            eligible[feature], errors="coerce"
        )

    X = eligible[FEATURES].copy()

    # Reject missing, infinite or invalid values rather than silently
    # imputing environmental measurements.
    invalid = ~np.isfinite(X.to_numpy(dtype=float))

    if invalid.any():
        bad_rows, bad_cols = np.where(invalid)
        problems = sorted({
            FEATURES[col] for col in bad_cols
        })
        raise ValueError(
            f"Missing/infinite feature values detected: {problems}. "
            "Investigate the source data before clustering."
        )

    # 7. Save the final modeling table
    modeling = eligible[["catchment_id"] + FEATURES].copy()

    modeling.to_csv(
        DATA_DIR / "modeling_dataset.csv",
        index=False,
    )

    # 8. Descriptive statistics and feature correlations
    X.describe().T.to_csv(
        DATA_DIR / "feature_summary.csv"
    )

    X.corr(method="pearson").to_csv(
        DATA_DIR / "feature_correlation.csv"
    )

    # 9. Print quality-control summary
    print("\n" + "=" * 65)
    print("DATASET QUALITY CHECK")
    print("=" * 65)

    print(f"Catchments before coverage filtering: {len(merged)}")
    print(f"Coverage threshold: {COVERAGE_THRESHOLD:.0%}")
    print(f"Excluded catchments: {len(excluded)}")
    print(f"Final modeling catchments: {len(modeling)}")
    print(f"Selected features: {len(FEATURES)}")
    print(f"Missing feature values: {int(X.isna().sum().sum())}")
    print(f"Duplicate catchment IDs: {modeling['catchment_id'].duplicated().sum()}")

    print("\nSelected features:")
    for i, feature in enumerate(FEATURES, start=1):
        print(f"{i:2}. {feature}")

    print("\nFiles created:")
    print("- catchment_features_merged_all.csv")
    print("- rainfall_coverage_exclusions.csv")
    print("- modeling_dataset.csv")
    print("- feature_summary.csv")
    print("- feature_correlation.csv")

    print("\nDataset preview:")
    print(modeling.head().to_string(index=False))

    print("\nDataset preparation completed successfully.")


if __name__ == "__main__":
    main()
