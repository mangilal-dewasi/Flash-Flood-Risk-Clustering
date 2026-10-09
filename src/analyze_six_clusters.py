
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/processed"

df = pd.read_csv(
    DATA_DIR / "clustering_dataset.csv",
    dtype={"catchment_id": str},
)

features = [c for c in df.columns if c != "catchment_id"]
X = df[features].apply(pd.to_numeric, errors="raise")

if not np.isfinite(X.to_numpy(dtype=float)).all():
    raise ValueError("Missing or infinite feature values found.")

# Standardize exactly as in the previous experiments
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit the six-cluster candidate
model = KMeans(
    n_clusters=6,
    random_state=42,
    n_init=20,
)
labels = model.fit_predict(X_scaled)

# Save catchment assignments
assignments = df[["catchment_id"]].copy()
assignments["cluster"] = labels
assignments.to_csv(
    DATA_DIR / "kmeans_k6_assignments.csv",
    index=False,
)

# Cluster means in original measurement units
original = df[features].copy()
original["cluster"] = labels

profiles = original.groupby("cluster")[features].mean()
profiles.insert(
    0, "catchment_count",
    original.groupby("cluster").size(),
)
profiles.to_csv(DATA_DIR / "kmeans_k6_profiles.csv")

# Standardized profiles help identify distinguishing features
z_df = pd.DataFrame(X_scaled, columns=features)
z_df["cluster"] = labels

z_profiles = z_df.groupby("cluster")[features].mean()
z_profiles.insert(
    0, "catchment_count",
    z_df.groupby("cluster").size(),
)
z_profiles.to_csv(DATA_DIR / "kmeans_k6_profiles_zscores.csv")

# Calculate per-catchment silhouette scores
scores = silhouette_samples(X_scaled, labels)

silhouette_df = pd.DataFrame({
    "catchment_id": df["catchment_id"],
    "cluster": labels,
    "silhouette": scores,
})
silhouette_df.to_csv(
    DATA_DIR / "kmeans_k6_silhouette_by_catchment.csv",
    index=False,
)

silhouette_summary = silhouette_df.groupby("cluster").agg(
    catchment_count=("silhouette", "size"),
    mean_silhouette=("silhouette", "mean"),
    median_silhouette=("silhouette", "median"),
    negative_fraction=("silhouette", lambda s: (s < 0).mean()),
)
silhouette_summary.to_csv(
    DATA_DIR / "kmeans_k6_silhouette_summary.csv"
)

print("=" * 65)
print("FLASH-FLOOD PATTERN ATLAS: SIX-CLUSTER ANALYSIS")
print("=" * 65)

print("\nOVERALL SILHOUETTE")
print(round(silhouette_score(X_scaled, labels), 4))

print("\nCLUSTER SIZES")
print(pd.Series(labels).value_counts().sort_index().to_string())

print("\nSTANDARDIZED CLUSTER PROFILES")
print(z_profiles.round(2).to_string())

print("\nSILHOUETTE BY CLUSTER")
print(silhouette_summary.round(3).to_string())

print("\nSaved six-cluster assignments, profiles and silhouette reports.")
