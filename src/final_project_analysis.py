"""
Flash-Flood Pattern Atlas: consolidated one-run analysis.

Run from repository root:
    python src/final_project_analysis.py

Required input: data/processed/clustering_dataset.csv
Optional input: data/processed/modeling_dataset.csv

Outputs: reports/final_analysis_report.txt, CSV summaries in data/processed/,
and three figures in reports/figures/. No follow-up analysis script is needed.
Requires pandas, numpy, scikit-learn, matplotlib.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, silhouette_samples, davies_bouldin_score,
    calinski_harabasz_score, adjusted_rand_score
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

    # Hierarchical comparison on the same standardised features and catchments.
    # Ward linkage is valid only with Euclidean distance; average linkage supports
    # Euclidean and Manhattan distances.
    hier_rows = []
    hier_assignments = []
    kmeans_reference = {}
    for k in [4, 5, 6]:
        km_labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(X)
        kmeans_reference[k] = km_labels
        methods = [
            ("Agglomerative Ward", "euclidean", "ward"),
            ("Agglomerative Average", "euclidean", "average"),
            ("Agglomerative Average", "manhattan", "average"),
        ]
        for method_name, metric, linkage in methods:
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric=metric)
            labels = model.fit_predict(X)
            sizes = pd.Series(labels).value_counts().sort_index()
            sil = float(silhouette_score(X, labels, metric=metric))
            db = float(davies_bouldin_score(X, labels))
            ch = float(calinski_harabasz_score(X, labels))
            ari = float(adjusted_rand_score(km_labels, labels))
            hier_rows.append({
                "k": k, "method": method_name, "distance": metric,
                "silhouette": sil, "davies_bouldin_euclidean": db,
                "calinski_harabasz_euclidean": ch,
                "smallest_cluster_size": int(sizes.min()),
                "largest_cluster_size": int(sizes.max()),
                "cluster_sizes": str({int(i): int(v) for i, v in sizes.items()}),
                "ari_vs_kmeans_same_k": ari,
            })
            for cid, label in zip(df["catchment_id"], labels):
                hier_assignments.append({
                    "catchment_id": cid, "method": method_name, "distance": metric,
                    "k": k, "cluster": int(label),
                })
    hier_df = pd.DataFrame(hier_rows)
    hier_df.to_csv(DATA / "final_hierarchical_comparison.csv", index=False)
    pd.DataFrame(hier_assignments).to_csv(DATA / "final_hierarchical_assignments.csv", index=False)
    log("\nHIERARCHICAL CLUSTERING COMPARISON")
    log(hier_df.round(4).to_string(index=False))
    log("ARI compares each hierarchical partition with K-Means at the same k; cluster labels are permutation-invariant.")
    log("Ward linkage uses Euclidean distance; Manhattan is evaluated with average linkage.")
    log("Davies-Bouldin and Calinski-Harabasz are Euclidean-space diagnostics, including for Manhattan partitions.")

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
    log("Working recommendation: retain K-Means k=6 as the leading candidate, with k=5 as a simpler sensitivity alternative.")
    log("Hierarchical comparison is supplementary: evaluate separation, stability, cluster sizes and interpretability together.")
    log("Do not label clusters Low/Medium/High flood risk: no independent flood outcomes were used.")
    log("Interpret clusters as exploratory rainfall–terrain patterns, not flood probabilities.")
    log("Any cluster with low or negative silhouette must be described as overlapping/weakly separated.")
    report = REPORTS / "final_analysis_report.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nDONE. One consolidated report: {report}")
    print("Send me only reports/final_analysis_report.txt. No extra analysis script is needed.")


if __name__ == "__main__":
    main()
