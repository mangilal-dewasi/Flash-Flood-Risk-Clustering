
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/processed/clustering_dataset.csv"
OUTPUT_DIR = ROOT / "data/processed"
PLOT_DIR = ROOT / "reports/figures"

ID_COLUMN = "catchment_id"
K_VALUES = range(2, 9)
RANDOM_STATE = 42

# --------------------------------------------------
# 1. Load and validate data
# --------------------------------------------------
df = pd.read_csv(DATA_PATH, dtype={ID_COLUMN: str})

if ID_COLUMN not in df.columns:
    raise ValueError(f"Missing ID column: {ID_COLUMN}")

feature_columns = [c for c in df.columns if c != ID_COLUMN]
X = df[feature_columns].apply(pd.to_numeric, errors="raise")

if len(df) < 9 or X.shape[1] < 2:
    raise ValueError("Insufficient rows or features for this experiment.")

if df[ID_COLUMN].duplicated().any():
    raise ValueError("Duplicate catchment IDs found.")

if X.isna().any().any():
    raise ValueError("Missing feature values found.")

if not X.map(lambda value: pd.notna(value) and abs(value) != float("inf")).all().all():
    raise ValueError("Infinite feature values found.")

print("=" * 60)
print("FLASH-FLOOD PATTERN ATLAS: K-MEANS EXPERIMENTS")
print("=" * 60)
print(f"Catchments: {len(df)}")
print(f"Features: {len(feature_columns)}")
print(f"Testing k values: {list(K_VALUES)}")

# --------------------------------------------------
# 2. Standardize features
# --------------------------------------------------
# Important: scaling is necessary because features use
# different units, such as millimetres, metres and degrees.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------------
# 3. Compare different cluster counts
# --------------------------------------------------
results = []
models = {}

for k in K_VALUES:
    model = KMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=20,
    )

    labels = model.fit_predict(X_scaled)
    models[k] = (model, labels)

    results.append({
        "k": k,
        "inertia": model.inertia_,
        "silhouette_score": silhouette_score(X_scaled, labels),
        "davies_bouldin_score": davies_bouldin_score(
            X_scaled, labels
        ),
        "calinski_harabasz_score": calinski_harabasz_score(
            X_scaled, labels
        ),
        "smallest_cluster_size": min(
            pd.Series(labels).value_counts()
        ),
    })

metrics = pd.DataFrame(results)

# Save all evaluation results
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
metrics.to_csv(OUTPUT_DIR / "kmeans_evaluation.csv", index=False)

# Highest silhouette is a candidate, not an automatic final decision.
best_k = int(
    metrics.loc[metrics["silhouette_score"].idxmax(), "k"]
)

best_model, best_labels = models[best_k]

# --------------------------------------------------
# 4. Save provisional cluster assignments
# --------------------------------------------------
assignments = df[[ID_COLUMN]].copy()
assignments["cluster"] = best_labels
assignments.to_csv(
    OUTPUT_DIR / "kmeans_cluster_assignments.csv",
    index=False,
)

# --------------------------------------------------
# 5. Describe clusters in original measurement units
# --------------------------------------------------
profile_data = X.copy()
profile_data["cluster"] = best_labels

profiles = profile_data.groupby("cluster")[feature_columns].mean()
profiles.insert(0, "catchment_count",
                profile_data.groupby("cluster").size())
profiles.to_csv(OUTPUT_DIR / "kmeans_cluster_profiles.csv")

# --------------------------------------------------
# 6. Plot the elbow curve
# --------------------------------------------------
PLOT_DIR.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(8, 5))
plt.plot(metrics["k"], metrics["inertia"], marker="o")
plt.xticks(list(K_VALUES))
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (within-cluster sum of squares)")
plt.title("K-Means Elbow Curve")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "kmeans_elbow_curve.png", dpi=200)
plt.close()

# --------------------------------------------------
# 7. Print results
# --------------------------------------------------
print("\nEVALUATION RESULTS")
print(metrics.round(4).to_string(index=False))

print(f"\nBest silhouette candidate: k = {best_k}")
print("\nCLUSTER SIZES")
print(assignments["cluster"].value_counts().sort_index().to_string())

print("\nOUTPUT FILES")
print("- data/processed/kmeans_evaluation.csv")
print("- data/processed/kmeans_cluster_assignments.csv")
print("- data/processed/kmeans_cluster_profiles.csv")
print("- reports/figures/kmeans_elbow_curve.png")

print(
    "\nNote: the highest-silhouette model is provisional. "
    "Review all metrics, cluster sizes and feature profiles "
    "before selecting the final model."
)
