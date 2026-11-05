"""
PCA Visualisation and Clustering selection page for Flash-Flood Pattern Atlas.
Integrate into the existing Streamlit app by importing render_objective2_page
and calling render_objective2_page(ROOT / "data" / "processed").

Expected files in data/processed:
  clustering_dataset.csv
  final_kmeans_evaluation.csv
  final_distance_method_comparison.csv
  final_k5_assignments.csv, final_k6_assignments.csv (optional; PCA falls back gracefully)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

FEATURE_LABELS = {
    "mean_monsoon_rainfall_mm": "Mean monsoon rainfall (mm)",
    "mean_heavy_rain_days_50mm": "Heavy-rain days ≥50 mm/year",
    "mean_very_heavy_rain_days_100mm": "Very-heavy-rain days ≥100 mm/year",
    "mean_max_3day_rainfall_mm": "Mean maximum 3-day rainfall (mm)",
    "rainfall_cv": "Rainfall coefficient of variation",
    "mean_elevation_m": "Mean elevation (m)",
    "relief_m": "Relief (m)",
    "mean_slope_deg": "Mean slope (degrees)",
    "slope_std_deg": "Slope standard deviation (degrees)",
}

def _read_csv(folder: Path, filename: str):
    path = folder / filename
    return pd.read_csv(path) if path.exists() else None

def render_objective2_page(data_dir):
    """Render the Objective 2 analysis UI. data_dir points to data/processed."""
    data_dir = Path(data_dir)

    # Project-wide header hiding preference: keep toolbar/header hidden.
    st.markdown(
        """<style>
        [data-testid="stHeader"], [data-testid="stToolbar"],
        [data-testid="stStatusWidget"], .stAppDeployButton {
            display: none !important;
        }
        .block-container { padding-top: 1.15rem !important; }
        </style>""",
        unsafe_allow_html=True,
    )

    st.markdown("## Clustering Selection")
    st.caption(
        "Compare candidate cluster counts and distance definitions. "
        "PCA is a 2-D inspection view; clustering is performed in the full standardised feature space."
    )
    st.warning(
        "Research boundary: clusters are exploratory rainfall–terrain patterns"
    )

    evaluation = _read_csv(data_dir, "final_kmeans_evaluation.csv")
    methods = _read_csv(data_dir, "final_distance_method_comparison.csv")
    dataset = _read_csv(data_dir, "clustering_dataset.csv")

    if evaluation is None:
        st.error("Could not find final_kmeans_evaluation.csv in data/processed.")
        return

    evaluation = evaluation.sort_values("k")
    k_options = evaluation["k"].astype(int).tolist()
    default_k = 6 if 6 in k_options else k_options[0]

    left, right = st.columns([1, 2])
    with left:
        selected_k = st.selectbox("Candidate cluster count (k)", k_options, index=k_options.index(default_k))
    with right:
        st.markdown(
            "**Metric reading guide** · WCSS / inertia: lower is better but always decreases with k. "
            "Silhouette: higher is better. Davies–Bouldin: lower is better. "
            "No single metric determines the answer."
        )

    row = evaluation.loc[evaluation["k"] == selected_k].iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("WCSS / inertia", f"{row['inertia']:.2f}", help="Sum of squared Euclidean distances to K-Means centroids.")
    m2.metric("Silhouette", f"{row['silhouette']:.4f}", help="Higher indicates stronger cohesion/separation under Euclidean distance.")
    m3.metric("Davies–Bouldin", f"{row['davies_bouldin']:.4f}", help="Lower indicates more compact, separated clusters.")
    m4.metric("Smallest cluster", int(row["smallest_cluster_size"]), help="A small cluster can represent a meaningful subgroup or a fragile partition.")

    st.markdown("### Validation across candidate k")
    metric_tabs = st.tabs(["WCSS / elbow", "Silhouette", "Davies–Bouldin"])
    with metric_tabs[0]:
        fig = px.line(evaluation, x="k", y="inertia", markers=True, title="Within-cluster sum of squares")
        fig.update_yaxes(title="WCSS / inertia")
        st.plotly_chart(fig, use_container_width=True)
    with metric_tabs[1]:
        fig = px.line(evaluation, x="k", y="silhouette", markers=True, title="Silhouette score (higher is better)")
        fig.update_yaxes(title="Silhouette")
        st.plotly_chart(fig, use_container_width=True)
    with metric_tabs[2]:
        fig = px.line(evaluation, x="k", y="davies_bouldin", markers=True, title="Davies–Bouldin index (lower is better)")
        fig.update_yaxes(title="Davies–Bouldin")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Distance-method comparison")
    if methods is None:
        st.info("Distance comparison CSV is not available; add final_distance_method_comparison.csv to data/processed.")
    else:
        st.dataframe(methods[methods["k"] == selected_k], use_container_width=True, hide_index=True)
        st.caption(
            "Euclidean K-Means optimises squared Euclidean distance to centroids. "
            "Manhattan comparison uses a medoid-based method (PAM-style K-Medoids), not Manhattan K-Means. "
            "Compare matching-distance silhouette values cautiously; Euclidean DB/CH columns are supplementary diagnostics."
        )

    st.markdown("### PCA separation inspection")
    if dataset is None:
        st.info("PCA view needs clustering_dataset.csv in data/processed.")
    else:
        id_col = "catchment_id"
        features = [c for c in dataset.columns if c != id_col]
        numeric = dataset[features].select_dtypes(include=np.number)
        clean = dataset[[id_col]].join(numeric).dropna()
        if len(clean) < 3 or numeric.shape[1] < 2:
            st.warning("Not enough complete numeric records/features for a 2-component PCA.")
        else:
            scaled = StandardScaler().fit_transform(clean[numeric.columns])
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(scaled)
            plot_df = pd.DataFrame({
                id_col: clean[id_col].astype(str).values,
                "PC1": coords[:, 0],
                "PC2": coords[:, 1],
            })
            assign_file = data_dir / f"final_k{selected_k}_assignments.csv"
            if assign_file.exists():
                assignments = pd.read_csv(assign_file)
                assignment_id = id_col if id_col in assignments.columns else assignments.columns[0]
                cluster_col = next((c for c in ["cluster", "cluster_label", "cluster_id"] if c in assignments.columns), None)
                if cluster_col:
                        # Convert both catchment IDs to strings before merging
                        plot_df[id_col] = plot_df[id_col].astype(str)

                        assignments[assignment_id] = assignments[assignment_id].astype(str)

                        plot_df = plot_df.merge(
                            assignments[[assignment_id, cluster_col]].rename(
                                columns={
                                    assignment_id: id_col,
                                    cluster_col: "Cluster"
                                }
                            ),
                            on=id_col,
                            how="left"
                        )
            if "Cluster" in plot_df:
                fig = px.scatter(plot_df, x="PC1", y="PC2", color=plot_df["Cluster"].astype(str),
                                 hover_name=id_col, title=f"PCA projection · k={selected_k}")
            else:
                fig = px.scatter(plot_df, x="PC1", y="PC2", hover_name=id_col,
                                 title="PCA projection (cluster assignments unavailable)")
            fig.update_layout(xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)",
                              yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"PC1 + PC2 explain {pca.explained_variance_ratio_.sum()*100:.1f}% of total variance. "
                "Visible overlap in 2-D does not prove overlap in the full feature space."
            )
