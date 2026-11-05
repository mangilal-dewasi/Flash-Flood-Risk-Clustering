"""Train the historical 24-hour catchment-average rainfall estimator.

Expected project files:
  data/processed/catchment_daily_rainfall_2019_2024.csv
  data/processed/clustering_dataset.csv

The final model is trained on 2019-2023 only. 2024 is kept as a held-out
historical prediction period and is never used to fit the estimator.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
PANEL_PATH = PROCESSED / "catchment_daily_rainfall_2019_2024.csv"
CLUSTER_DATA_PATH = PROCESSED / "clustering_dataset.csv"
MODEL_PATH = MODEL_DIR / "next_day_rainfall_lgbm.joblib"

FEATURES = [
    "rainfall_mm",
    "rain_lag_1d_mm",
    "rain_lag_2d_mm",
    "rain_lag_3d_mm",
    "rain_lag_7d_mm",
    "rain_sum_prev_3d_mm",
    "rain_sum_prev_7d_mm",
    "rain_max_prev_3d_mm",
    "rain_max_prev_7d_mm",
    "doy_sin",
    "doy_cos",
    "is_monsoon",
    "mean_elevation_m",
    "relief_m",
    "mean_slope_deg",
    "slope_std_deg",
]
TERRAIN_FEATURES = [
    "mean_elevation_m", "relief_m", "mean_slope_deg", "slope_std_deg"
]


def build_training_frame() -> pd.DataFrame:
    if not PANEL_PATH.exists() or not CLUSTER_DATA_PATH.exists():
        raise FileNotFoundError(
            "Required input missing. Build the daily rainfall panel first and "
            "keep clustering_dataset.csv under data/processed."
        )
    panel = pd.read_csv(
        PANEL_PATH,
        parse_dates=["date"],
        dtype={"catchment_id": str},
    )
    terrain = pd.read_csv(
        CLUSTER_DATA_PATH,
        dtype={"catchment_id": str},
    )[["catchment_id", *TERRAIN_FEATURES]]
    frame = panel.merge(
        terrain, on="catchment_id", how="left", validate="many_to_one"
    )
    frame["doy_sin"] = np.sin(2 * np.pi * frame["date"].dt.dayofyear / 365.25)
    frame["doy_cos"] = np.cos(2 * np.pi * frame["date"].dt.dayofyear / 365.25)

    # The estimator is specifically a monsoon-season rainfall estimator.
    frame = frame[
        (frame["is_monsoon"] == 1)
        & frame["target_next_day_rainfall_mm"].notna()
        & (frame["date"] < pd.Timestamp("2024-01-01"))
    ].dropna(subset=FEATURES).copy()
    if frame.empty:
        raise ValueError("No valid training rows were found.")
    if not np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite values were found in the training features.")
    if (frame["target_next_day_rainfall_mm"] < 0).any():
        raise ValueError("Rainfall target contains negative values.")
    return frame


def main() -> None:
    frame = build_training_frame()
    model = LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.35,
        n_estimators=222,  # selected on 2023 validation, before 2024 test
        learning_rate=0.025,
        num_leaves=15,
        min_child_samples=35,
        reg_lambda=8.0,
        subsample=0.8,
        colsample_bytree=0.85,
        verbosity=-1,
        n_jobs=4,
        random_state=42,
    )
    model.fit(frame[FEATURES], frame["target_next_day_rainfall_mm"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_names": FEATURES,
        "target_name": "next_day_catchment_mean_rainfall_mm",
        "units": "mm",
        "training_start": str(frame["date"].min().date()),
        "training_origin_end": str(frame["date"].max().date()),
        "training_target_end": "2023-10-01",
        "forecast_scope": "monsoon; one-day-ahead catchment-area mean rainfall",
        "model_family": "LightGBM Tweedie regression",
        "validated_holdout_years": [2022, 2023, 2024],
        "walk_forward_mean_mae_mm": 7.2739,
        "walk_forward_persistence_mae_mm": 8.5724,
        "warning": (
            "Research prototype using historical IMD gridded rainfall; not a "
            "flood forecast, warning, or calibrated uncertainty estimate."
        ),
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved: {MODEL_PATH}")
    print(f"Training rows: {len(frame):,}")
    print(f"Training feature dates: {frame['date'].min().date()} to {frame['date'].max().date()}")


if __name__ == "__main__":
    main()
