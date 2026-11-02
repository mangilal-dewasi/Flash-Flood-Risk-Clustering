
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed"

DATA_PATH = DATA_DIR / "clustering_dataset.csv"
ASSIGNMENTS_PATH = DATA_DIR / "kmeans_cluster_assignments.csv"

df = pd.read_csv(DATA_PATH, dtype={"catchment_id": str})
assignments = pd.read_csv(
    ASSIGNMENTS_PATH, dtype={"catchment_id": str}
)

features = [c for c in df.columns if c != "catchment_id"]

data = df.merge(
    assignments[["catchment_id", "cluster"]],
    on="catchment_id",
    how="left",
    validate="one_to_one",
)

if data["cluster"].isna().any():
    raise ValueError("Some catchments have no cluster assignment.")

X = data[features].apply(pd.to_numeric, errors="raise")

if not np.isfinite(X.to_numpy(dtype=float)).all():
    raise ValueError("Missing or infinite feature values found.")

labels = data["cluster"].astype(int).to_numpy()

# Recreate the same full-dataset standardization used for K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cluster means in original units
original_profiles = data.groupby("cluster")[features].mean()
original_profiles.insert(
    0, "catchment_count",
    data.groupby("cluster").size()
)

original_profiles.to_csv(
    DATA_DIR / "cluster_profiles_original_units.csv"
)

# Cluster means in standardized units:
# positive = above overall average; negative = below overall average
z_df = pd.DataFrame(X_scaled, columns=features)
z_df["cluster"] = labels

z_profiles = z_df.groupby("cluster").mean()
z_profiles.insert(
    0, "catchment_count",
    pd.Series(labels).value_counts().sort_index()
)

z_profiles.to_csv(DATA_DIR / "cluster_profiles_zscores.csv")

# Individual catchment silhouette values
sample_scores = silhouette_samples(X_scaled, labels)

silhouette_data = pd.DataFrame({
    "cluster": labels,
    "silhouette": sample_scores,
})

silhouette_summary = silhouette_data.groupby("cluster").agg(
    catchment_count=("silhouette", "size"),
    mean_silhouette=("silhouette", "mean"),
    median_silhouette=("silhouette", "median"),
    negative_silhouette_fraction=(
        "silhouette", lambda s: (s < 0).mean()
    ),
)

silhouette_summary.to_csv(
    DATA_DIR / "cluster_silhouette_summary.csv"
)

print("=" * 65)
print("CLUSTER INTERPRETATION: PROVISIONAL K-MEANS MODEL")
print("=" * 65)

print("\nCLUSTER SIZES")
print(data["cluster"].value_counts().sort_index().to_string())

print("\nSTANDARDIZED CLUSTER PROFILES (Z-SCORES)")
print(z_profiles.round(2).to_string())

print("\nSILHOUETTE QUALITY BY CLUSTER")
print(silhouette_summary.round(3).to_string())

print("\nSaved:")
print("- cluster_profiles_original_units.csv")
print("- cluster_profiles_zscores.csv")
print("- cluster_silhouette_summary.csv")

print(
    "\nInterpretation guide: positive z-scores indicate feature "
    "means above the overall dataset average; negative scores "
    "indicate below-average means. Negative silhouette values "
    "suggest a catchment may fit another cluster better."
)
