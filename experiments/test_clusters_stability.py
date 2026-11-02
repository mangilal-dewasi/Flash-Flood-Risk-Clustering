
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    davies_bouldin_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/clustering_dataset.csv"
OUT_DIR = ROOT / "data/processed"

SEEDS = [0, 1, 7, 21, 42, 73, 100, 123, 2026, 999]
K_VALUES = [4, 5, 6]

df = pd.read_csv(DATA_PATH, dtype={"catchment_id": str})
features = [c for c in df.columns if c != "catchment_id"]
X = df[features].apply(pd.to_numeric, errors="raise")

if not np.isfinite(X.to_numpy(dtype=float)).all():
    raise ValueError("Missing or infinite values found.")

# Same standardization approach as the first experiment
X_scaled = StandardScaler().fit_transform(X)

metric_rows = []
stability_rows = []

for k in K_VALUES:
    labels_by_seed = {}

    for seed in SEEDS:
        model = KMeans(
            n_clusters=k,
            random_state=seed,
            n_init=20,
        )
        labels = model.fit_predict(X_scaled)
        labels_by_seed[seed] = labels

        metric_rows.append({
            "k": k,
            "seed": seed,
            "silhouette": silhouette_score(X_scaled, labels),
            "davies_bouldin": davies_bouldin_score(X_scaled, labels),
            "inertia": model.inertia_,
            "smallest_cluster_size": int(
                pd.Series(labels).value_counts().min()
            ),
        })

    # Compare each seed with the seed-42 reference partition
    reference = labels_by_seed[42]
    for seed, labels in labels_by_seed.items():
        stability_rows.append({
            "k": k,
            "seed": seed,
            "ARI_vs_seed42": adjusted_rand_score(reference, labels),
        })

    # Compare all distinct pairs of seeds
    pair_scores = []
    for i, seed_a in enumerate(SEEDS):
        for seed_b in SEEDS[i + 1:]:
            pair_scores.append(
                adjusted_rand_score(
                    labels_by_seed[seed_a],
                    labels_by_seed[seed_b],
                )
            )

    print(f"\nk={k}")
    print(
        "Mean pairwise Adjusted Rand Index:",
        round(float(np.mean(pair_scores)), 4),
    )
    print(
        "Minimum pairwise Adjusted Rand Index:",
        round(float(np.min(pair_scores)), 4),
    )

metrics = pd.DataFrame(metric_rows)
stability = pd.DataFrame(stability_rows)

metrics.to_csv(OUT_DIR / "kmeans_seed_metrics.csv", index=False)
stability.to_csv(OUT_DIR / "kmeans_seed_stability.csv", index=False)

summary = metrics.groupby("k").agg(
    silhouette_mean=("silhouette", "mean"),
    silhouette_std=("silhouette", "std"),
    davies_bouldin_mean=("davies_bouldin", "mean"),
    inertia_std=("inertia", "std"),
    smallest_cluster_min=("smallest_cluster_size", "min"),
)

print("\nMETRIC SUMMARY ACROSS SEEDS")
print(summary.round(4).to_string())

print("\nARI VS SEED 42")
print(
    stability.pivot(
        index="seed", columns="k", values="ARI_vs_seed42"
    ).round(4).to_string()
)

print("\nSaved:")
print("- data/processed/kmeans_seed_metrics.csv")
print("- data/processed/kmeans_seed_stability.csv")
