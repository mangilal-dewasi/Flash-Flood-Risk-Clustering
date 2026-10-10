"""
Flash-Flood Pattern Atlas: consolidated one-run analysis.

Run from repository root:
    python src/final_project_analysis.py

Required input: data/processed/clustering_dataset.csv
Optional input: data/processed/modeling_dataset.csv

Outputs: reports/final_analysis_report.txt, CSV summaries in data/processed/,
and figures in reports/figures/. No follow-up analysis script is needed.
Requires pandas, numpy, scikit-learn, matplotlib.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, silhouette_samples, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score, pairwise_distances
)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
FEATURE_FILE = DATA / "clustering_dataset.csv"
SEEDS = [0, 1, 7, 21, 42, 73, 100, 123, 2026, 999]


def find_id(df):
    for col in ["catchment_id", "HYBAS_ID", "hybas_id", "id", "COMID"]:
        if col in df.columns:
            return col
    return None


def pam_kmedoids_manhattan(X, n_clusters, max_iter=100):
    """Small-data PAM-style K-Medoids using Manhattan distances.

    Greedy BUILD initialization followed by best-improving single medoid swaps.
    Returns integer labels, medoid row indices, and total Manhattan-distance cost.
    Suitable for this project dataset (about 113 catchments), not very large data.
    """
    D = pairwise_distances(X, metric="manhattan")
    n = D.shape[0]
    if not 2 <= n_clusters < n:
        raise ValueError("n_clusters must be between 2 and n_samples - 1")

    # BUILD: choose the first medoid minimizing total distance, then greedily add.
    first = int(np.argmin(D.sum(axis=1)))
    medoids = [first]
    nearest = D[:, first].copy()
    while len(medoids) < n_clusters:
        best_candidate, best_cost = None, np.inf
        for candidate in range(n):
            if candidate in medoids:
                continue
            cost = np.minimum(nearest, D[:, candidate]).sum()
            if cost < best_cost:
                best_candidate, best_cost = candidate, cost
        medoids.append(int(best_candidate))
        nearest = np.minimum(nearest, D[:, best_candidate])

    medoids = np.array(medoids, dtype=int)
    current_cost = D[:, medoids].min(axis=1).sum()
    for _ in range(max_iter):
        best_swap, best_cost = None, current_cost
        medoid_set = set(medoids.tolist())
        for old in medoids:
            for candidate in range(n):
                if candidate in medoid_set:
                    continue
                trial = medoids.copy()
                trial[np.where(trial == old)[0][0]] = candidate
                cost = D[:, trial].min(axis=1).sum()
                if cost < best_cost - 1e-10:
                    best_cost = cost
                    best_swap = (int(old), int(candidate))
        if best_swap is None:
            break
        old, new = best_swap
        medoids[np.where(medoids == old)[0][0]] = new
        current_cost = best_cost

    distances_to_medoids = D[:, medoids]
    labels = distances_to_medoids.argmin(axis=1)
    return labels, medoids, float(distances_to_medoids.min(axis=1).sum())

def main():
    DATA.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Missing {FEATURE_FILE}. Put this script in src/ and run it "
            "from the project repository."
        )

    lines = []
    def log(s=""):
        s = str(s)
        print(s)
        lines.append(s)

    df = pd.read_csv(FEATURE_FILE)
    id_col = find_id(df)
    if id_col is None:
        raise ValueError(f"No catchment ID column found. Columns: {df.columns.tolist()}")
    df = df.rename(columns={id_col: "catchment_id"})
    features = [
        c for c in df.columns
        if c != "catchment_id" and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(features) < 2:
        raise ValueError(f"Need at least two numeric features. Found: {features}")

    numeric = df[features].replace([np.inf, -np.inf], np.nan)
    valid = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    dropped = int((~valid).sum())
    df = df.loc[valid].reset_index(drop=True)
    Xraw = df[features].astype(float)
    X = StandardScaler().fit_transform(Xraw)

    log("=" * 75)
    log("FLASH-FLOOD PATTERN ATLAS — CONSOLIDATED FINAL ANALYSIS")
    log("=" * 75)
    log(f"Input: {FEATURE_FILE.relative_to(ROOT)}")
    log(f"Valid catchments: {len(df)}; excluded rows with non-finite values: {dropped}")
    log(f"Numeric features ({len(features)}): {', '.join(features)}")
    log(f"Duplicate IDs: {df['catchment_id'].duplicated().sum()}")
    log(f"Missing IDs: {df['catchment_id'].isna().sum()}")
    log(f"Missing/non-finite feature cells after filtering: {int((~np.isfinite(Xraw.to_numpy())).sum())}")

    # Feature summary and correlation analysis.
    Xraw.describe().T.to_csv(DATA / "final_feature_summary.csv")
    corr = Xraw.corr()
    corr.to_csv(DATA / "final_feature_correlation.csv")
    pairs = []
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            pairs.append((a, b, float(corr.loc[a, b]), abs(float(corr.loc[a, b]))))
    pairs.sort(key=lambda x: x[3], reverse=True)
    corr_df = pd.DataFrame(pairs, columns=["feature_1", "feature_2", "correlation", "absolute_correlation"])
    corr_df.to_csv(DATA / "final_high_feature_correlations.csv", index=False)
    log("\nTOP FEATURE CORRELATIONS")
    log(corr_df.head(12).round(3).to_string(index=False))

    # Evaluate k=2..8.
    eval_rows, models, labels_by_k = [], {}, {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = km.fit_predict(X)
        models[k], labels_by_k[k] = km, labels
        counts = pd.Series(labels).value_counts()
        eval_rows.append({
            "k": k, "inertia": km.inertia_,
            "silhouette": silhouette_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
            "smallest_cluster_size": int(counts.min()),
            "largest_cluster_size": int(counts.max())
        })
    evaluation = pd.DataFrame(eval_rows)
    evaluation.to_csv(DATA / "final_kmeans_evaluation.csv", index=False)
    log("\nK-MEANS COMPARISON (k=2..8)")
    log(evaluation.round(4).to_string(index=False))
    log(f"Best silhouette in sweep: k={int(evaluation.loc[evaluation.silhouette.idxmax(), 'k'])}")
    log(f"Lowest Davies-Bouldin in sweep: k={int(evaluation.loc[evaluation.davies_bouldin.idxmin(), 'k'])}")

    # Stability: pairwise ARI over all seed-pair combinations.
    stability_rows, seed_rows = [], []
    seed_labels_all = {}
    for k in [4, 5, 6]:
        labels_for_seeds = {}
        for seed in SEEDS:
            km = KMeans(n_clusters=k, random_state=seed, n_init=20)
            lab = km.fit_predict(X)
            labels_for_seeds[seed] = lab
            counts = pd.Series(lab).value_counts()
            seed_rows.append({
                "k": k, "seed": seed,
                "silhouette": silhouette_score(X, lab),
                "davies_bouldin": davies_bouldin_score(X, lab),
                "inertia": km.inertia_,
                "smallest_cluster_size": int(counts.min())
            })
        aris = [
            adjusted_rand_score(labels_for_seeds[a], labels_for_seeds[b])
            for i, a in enumerate(SEEDS) for b in SEEDS[i + 1:]
        ]
        group = [r for r in seed_rows if r["k"] == k]
        stability_rows.append({
            "k": k, "seed_count": len(SEEDS),
            "mean_pairwise_ARI": np.mean(aris), "min_pairwise_ARI": np.min(aris),
            "silhouette_mean": np.mean([r["silhouette"] for r in group]),
            "silhouette_std": np.std([r["silhouette"] for r in group]),
            "davies_bouldin_mean": np.mean([r["davies_bouldin"] for r in group]),
            "smallest_cluster_min": min(r["smallest_cluster_size"] for r in group)
        })
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(DATA / "final_seed_stability_summary.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(DATA / "final_seed_metrics.csv", index=False)
    log("\nRANDOM-SEED STABILITY")
    log(stability.round(4).to_string(index=False))

    # Detailed profiles, assignments and silhouette summaries for k=5 and k=6.
    all_profiles, all_sil, all_assignments = [], [], []
    for k in [5, 6]:
        lab = labels_by_k[k]
        sil = silhouette_samples(X, lab)
        assignment = pd.DataFrame({
            "catchment_id": df["catchment_id"], "cluster": lab,
            "silhouette": sil, "k": k
        })
        assignment.to_csv(DATA / f"final_k{k}_assignments.csv", index=False)
        all_assignments.append(assignment)

        original = Xraw.copy()
        original["cluster"] = lab
        prof = original.groupby("cluster")[features].mean()
        prof.insert(0, "catchment_count", original.groupby("cluster").size())
        prof = prof.reset_index()
        prof["k"] = k
        all_profiles.append(prof)
        prof.to_csv(DATA / f"final_k{k}_profiles_original_units.csv", index=False)

        zdf = pd.DataFrame(X, columns=features)
        zdf["cluster"] = lab
        zprof = zdf.groupby("cluster")[features].mean()
        zprof.insert(0, "catchment_count", zdf.groupby("cluster").size())
        zprof = zprof.reset_index()
        zprof.to_csv(DATA / f"final_k{k}_profiles_zscores.csv", index=False)

        sframe = pd.DataFrame({"cluster": lab, "silhouette": sil})
        ss = sframe.groupby("cluster").agg(
            catchment_count=("silhouette", "size"),
            mean_silhouette=("silhouette", "mean"),
            median_silhouette=("silhouette", "median"),
            negative_fraction=("silhouette", lambda s: float((s < 0).mean())),
            minimum_silhouette=("silhouette", "min")
        ).reset_index()
        ss["k"] = k
        all_sil.append(ss)
        ss.to_csv(DATA / f"final_k{k}_silhouette_by_cluster.csv", index=False)

        log(f"\nK={k}: overall silhouette = {silhouette_score(X, lab):.4f}")
        log("Cluster sizes: " + str(pd.Series(lab).value_counts().sort_index().to_dict()))
        log("Cluster silhouette summary:")
        log(ss.round(3).to_string(index=False))
        log("Cluster feature means in original units:")
        log(prof.round(3).to_string(index=False))
        log("Cluster feature means in standardized z-scores:")
        log(zprof.round(3).to_string(index=False))
        log(f"Lowest silhouette catchments for k={k}:")
        log(assignment.sort_values("silhouette")[["catchment_id", "cluster", "silhouette"]].head(10).round(4).to_string(index=False))
        log(f"Negative silhouette observations: {(sil < 0).sum()} / {len(sil)}")

    pd.concat(all_profiles, ignore_index=True).to_csv(DATA / "final_cluster_profiles_both_k.csv", index=False)
    pd.concat(all_sil, ignore_index=True).to_csv(DATA / "final_cluster_silhouette_summaries.csv", index=False)
    pd.concat(all_assignments, ignore_index=True).to_csv(DATA / "final_assignments_k5_k6.csv", index=False)

    # Optional all-features sensitivity check.
    full_path = DATA / "modeling_dataset.csv"
    if full_path.exists():
        full = pd.read_csv(full_path)
        fid = find_id(full)
        if fid:
            full = full.rename(columns={fid: "catchment_id"})
            ff = [c for c in full.columns if c != "catchment_id" and pd.api.types.is_numeric_dtype(full[c])]
            full[ff] = full[ff].replace([np.inf, -np.inf], np.nan)
            full = full.dropna(subset=ff)
            if len(ff) >= 2 and len(full) > 6:
                xf = StandardScaler().fit_transform(full[ff].astype(float))
                sens = []
                for k in [5, 6]:
                    ll = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(xf)
                    sens.append({
                        "feature_set": "all_numeric_features", "feature_count": len(ff),
                        "catchments": len(full), "k": k,
                        "silhouette": silhouette_score(xf, ll),
                        "davies_bouldin": davies_bouldin_score(xf, ll),
                        "smallest_cluster_size": int(pd.Series(ll).value_counts().min())
                    })
                sens_df = pd.DataFrame(sens)
                sens_df.to_csv(DATA / "final_feature_set_sensitivity.csv", index=False)
                log("\nOPTIONAL ALL-FEATURE SENSITIVITY CHECK")
                log(sens_df.round(4).to_string(index=False))
    else:
        log("\nOptional modeling_dataset.csv absent; skipped all-feature sensitivity check.")

    # Distance-method comparison: standard K-Means is Euclidean by design.
    # Compare it with a PAM-style K-Medoids model using Manhattan distances.
    # Silhouette is computed with each method's own distance geometry; DBI/CH
    # are supplementary Euclidean-space diagnostics and are not directly
    # equivalent to the Manhattan objective.
    distance_rows = []
    distance_assignments = []
    for k in [5, 6]:
        km_labels = labels_by_k[k]
        distance_rows.append({
            "k": k, "method": "K-Means", "distance": "Euclidean",
            "objective_name": "within-cluster sum of squared Euclidean distances (inertia)",
            "objective_value": float(models[k].inertia_),
            "silhouette_matching_distance": float(silhouette_score(X, km_labels, metric="euclidean")),
            "davies_bouldin_euclidean_diagnostic": float(davies_bouldin_score(X, km_labels)),
            "calinski_harabasz_euclidean_diagnostic": float(calinski_harabasz_score(X, km_labels)),
            "smallest_cluster_size": int(pd.Series(km_labels).value_counts().min()),
            "largest_cluster_size": int(pd.Series(km_labels).value_counts().max())
        })
        for cluster_id, label in enumerate(km_labels):
            distance_assignments.append({"catchment_id": df.loc[cluster_id, "catchment_id"], "k": k,
                                         "method": "K-Means", "distance": "Euclidean", "cluster": int(label)})

        med_labels, medoid_indices, manhattan_cost = pam_kmedoids_manhattan(X, k)
        counts = pd.Series(med_labels).value_counts()
        distance_rows.append({
            "k": k, "method": "K-Medoids (PAM-style)", "distance": "Manhattan",
            "objective_name": "sum of Manhattan distances to assigned medoids",
            "objective_value": manhattan_cost,
            "silhouette_matching_distance": float(silhouette_score(X, med_labels, metric="manhattan")),
            "davies_bouldin_euclidean_diagnostic": float(davies_bouldin_score(X, med_labels)),
            "calinski_harabasz_euclidean_diagnostic": float(calinski_harabasz_score(X, med_labels)),
            "smallest_cluster_size": int(counts.min()),
            "largest_cluster_size": int(counts.max())
        })
        for row_idx, label in enumerate(med_labels):
            distance_assignments.append({"catchment_id": df.loc[row_idx, "catchment_id"], "k": k,
                                         "method": "K-Medoids (PAM-style)", "distance": "Manhattan", "cluster": int(label)})
        pd.DataFrame({"medoid_index": medoid_indices,
                      "catchment_id": df.loc[medoid_indices, "catchment_id"].astype(str).to_numpy()}).to_csv(
                          DATA / f"final_kmedoids_manhattan_k{k}_medoids.csv", index=False)

    distance_comparison = pd.DataFrame(distance_rows)
    distance_comparison.to_csv(DATA / "final_distance_method_comparison.csv", index=False)
    pd.DataFrame(distance_assignments).to_csv(DATA / "final_distance_method_assignments.csv", index=False)
    log("\nDISTANCE-METHOD COMPARISON (K-Means Euclidean vs K-Medoids Manhattan)")
    log(distance_comparison.round(4).to_string(index=False))
    log("Note: objective values are method-specific and must not be compared numerically across methods.")
    log("Silhouette uses each method's stated distance. Davies-Bouldin and Calinski-Harabasz are Euclidean diagnostics only.")
    log("K-Medoids implementation: greedy BUILD initialization + best-improving medoid swaps; results are heuristic, not a proof of global optimum.")

    # PCA is used only to visualise the standardised feature space.
    # Clustering itself remains fitted on all selected standardised features.
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    pca_variance = pd.DataFrame({
        "component": ["PC1", "PC2"],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "explained_variance_percent": pca.explained_variance_ratio_ * 100,
    })
    pca_variance.to_csv(DATA / "final_pca_explained_variance.csv", index=False)
    pca_loadings = pd.DataFrame(
        pca.components_.T, index=features, columns=["PC1_loading", "PC2_loading"]
    ).rename_axis("feature").reset_index()
    pca_loadings.to_csv(DATA / "final_pca_loadings.csv", index=False)
    log("\nPCA VISUALISATION (two components; clustering still uses all features)")
    log(pca_variance.round(4).to_string(index=False))
    log(f"Combined variance explained by PC1 and PC2: {pca.explained_variance_ratio_.sum() * 100:.2f}%")
    log("PCA loadings saved to data/processed/final_pca_loadings.csv")

    # Figures.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for y, filename, title, ylabel in [
            ("inertia", "final_elbow_curve.png", "K-Means Elbow Curve", "Inertia (WCSS)"),
            ("silhouette", "final_silhouette_by_k.png", "Silhouette by Cluster Count", "Silhouette score")
        ]:
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            ax.plot(evaluation["k"], evaluation[y], marker="o")
            ax.set(title=title, xlabel="Number of clusters (k)", ylabel=ylabel)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(FIGURES / filename, dpi=180)
            plt.close(fig)
        # Separate PCA scatter plots make the k=5 and k=6 solutions easy to compare.
        for k in [5, 6]:
            labels = labels_by_k[k]
            fig, ax = plt.subplots(figsize=(8, 6))
            for cluster_id in sorted(np.unique(labels)):
                mask = labels == cluster_id
                ax.scatter(
                    X_pca[mask, 0], X_pca[mask, 1],
                    label=f"Cluster {cluster_id} (n={int(mask.sum())})",
                    alpha=0.8, s=38, edgecolors="none"
                )
            ax.set_title(
                f"PCA projection of catchments — K={k}\n"
                f"PC1: {pca.explained_variance_ratio_[0]*100:.1f}% | "
                f"PC2: {pca.explained_variance_ratio_[1]*100:.1f}% variance"
            )
            ax.set_xlabel("Principal Component 1")
            ax.set_ylabel("Principal Component 2")
            ax.axhline(0, linewidth=0.6, alpha=0.35)
            ax.axvline(0, linewidth=0.6, alpha=0.35)
            ax.grid(True, alpha=0.2)
            ax.legend(title="K-Means group", fontsize=8)
            fig.tight_layout()
            fig.savefig(FIGURES / f"final_pca_k{k}.png", dpi=180)
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(stability["k"], stability["mean_pairwise_ARI"], marker="o")
        ax.set(title="K-Means Stability Across Random Seeds", xlabel="k",
               ylabel="Mean pairwise Adjusted Rand Index", ylim=(0, 1.05))
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIGURES / "final_seed_stability.png", dpi=180)
        plt.close(fig)
        log("\nFigures saved in reports/figures/.")
    except Exception as exc:
        log(f"\nFigure generation skipped: {exc}")

    log("\nFINAL INTERPRETATION")
    stab = stability.set_index("k")
    ev = evaluation.set_index("k")
    for k in [5, 6]:
        log(f"k={k}: silhouette={ev.loc[k, 'silhouette']:.4f}; "
            f"DB={ev.loc[k, 'davies_bouldin']:.4f}; "
            f"mean pairwise ARI={stab.loc[k, 'mean_pairwise_ARI']:.4f}; "
            f"min pairwise ARI={stab.loc[k, 'min_pairwise_ARI']:.4f}")
    log("Working recommendation: retain k=6 as the leading candidate, then report the evidence and caveats.")
    log("Do not label clusters Low/Medium/High flood risk: no independent flood outcomes were used.")
    log("Interpret clusters as exploratory rainfall–terrain patterns, not flood probabilities.")
    log("Any cluster with low or negative silhouette must be described as overlapping/weakly separated.")
    report = REPORTS / "final_analysis_report.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nDONE. One consolidated report: {report}")
    print("Send me only reports/final_analysis_report.txt. No extra analysis script is needed.")


if __name__ == "__main__":
    main()
