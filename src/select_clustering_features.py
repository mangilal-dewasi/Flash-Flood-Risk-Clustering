
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data/processed/modeling_dataset.csv"
OUTPUT = ROOT / "data/processed/clustering_dataset.csv"
CORR_OUTPUT = ROOT / "data/processed/clustering_feature_correlation.csv"

FEATURES = [
    "mean_monsoon_rainfall_mm",
    "mean_heavy_rain_days_50mm",
    "mean_very_heavy_rain_days_100mm",
    "mean_max_3day_rainfall_mm",
    "rainfall_cv",
    "mean_elevation_m",
    "relief_m",
    "mean_slope_deg",
    "slope_std_deg",
]

# Load the original validated dataset
df = pd.read_csv(INPUT, dtype={"catchment_id": str})

# Check required columns
required = ["catchment_id"] + FEATURES
missing = [column for column in required if column not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Keep only the selected features and catchment ID
selected = df[required].copy()

# Check duplicate catchment IDs
if selected["catchment_id"].duplicated().any():
    raise ValueError("Duplicate catchment IDs found.")

# Convert features to numeric values
selected[FEATURES] = selected[FEATURES].apply(
    pd.to_numeric, errors="raise"
)

# Check missing or infinite values
if not np.isfinite(selected[FEATURES].to_numpy(dtype=float)).all():
    raise ValueError(
        "Missing or infinite values found in selected features."
    )

# Save selected dataset
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
selected.to_csv(OUTPUT, index=False)

# Calculate feature correlations
corr = selected[FEATURES].corr(method="pearson")
corr.to_csv(CORR_OUTPUT)

# Display quality-check results
print("=" * 55)
print("FLASH-FLOOD PATTERN ATLAS: FEATURE SELECTION")
print("=" * 55)

print(f"Catchments: {len(selected)}")
print(f"Selected features: {len(FEATURES)}")
print(f"Missing feature values: {selected[FEATURES].isna().sum().sum()}")
print(f"Duplicate catchment IDs: {selected['catchment_id'].duplicated().sum()}")

print("\nSaved files:")
print(f"- {OUTPUT}")
print(f"- {CORR_OUTPUT}")

print("\nSelected features:")
for i, feature in enumerate(FEATURES, start=1):
    print(f"{i}. {feature}")

print("\nFeature selection completed successfully!")
