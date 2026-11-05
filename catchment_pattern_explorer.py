from datetime import date, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


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

FORECAST_FEATURES = [
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
    "mean_elevation_m",
    "relief_m",
    "mean_slope_deg",
    "slope_std_deg",
]


@st.cache_data

def _load_dataset(csv_path: str) -> pd.DataFrame:
    data = pd.read_csv(csv_path, dtype={"catchment_id": str})
    missing = [column for column in ["catchment_id", *FEATURES] if column not in data.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    data[FEATURES] = data[FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(data[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("The catchment dataset contains missing or infinite feature values.")
    if data["catchment_id"].duplicated().any():
        raise ValueError("Duplicate catchment IDs were found.")
    return data


@st.cache_resource

def _fit_reference_model(csv_path: str):
    data = _load_dataset(csv_path)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(data[FEATURES])
    model = KMeans(n_clusters=6, random_state=42, n_init=20)
    labels = model.fit_predict(x_scaled)
    return scaler, model, labels, x_scaled


@st.cache_data

def _load_daily_panel(csv_path: str) -> pd.DataFrame:
    panel = pd.read_csv(
        csv_path,
        parse_dates=["date"],
        dtype={"catchment_id": str},
    )
    required = [
        "catchment_id", "date", "rainfall_mm", "rain_lag_1d_mm",
        "rain_lag_2d_mm", "rain_lag_3d_mm", "rain_lag_7d_mm",
        "rain_sum_prev_3d_mm", "rain_sum_prev_7d_mm",
        "rain_max_prev_3d_mm", "rain_max_prev_7d_mm",
        "is_monsoon", "target_next_day_rainfall_mm",
    ]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise ValueError(f"Daily rainfall panel is missing columns: {missing}")
    panel["catchment_id"] = panel["catchment_id"].astype(str)
    panel = panel.sort_values(["catchment_id", "date"]).reset_index(drop=True)
    return panel


@st.cache_resource

def _load_forecast_artifact(model_path: str):
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError("Forecast model artifact has an unexpected format.")
    expected = artifact.get("feature_names", [])
    if expected != FORECAST_FEATURES:
        raise ValueError("Forecast model feature order does not match this Explorer version.")
    return artifact


def _render_pattern_matching(data: pd.DataFrame, scaler, model, labels, x_scaled) -> None:
    st.write(
        "Enter rainfall and terrain characteristics to find the closest pattern "
        "among the catchments used in the existing six-cluster analysis."
    )

    medians = data[FEATURES].median()
    ranges = data[FEATURES].agg(["min", "max"])
    with st.form("catchment_pattern_form"):
        st.markdown("#### Enter catchment characteristics")
        columns = st.columns(3)
        entered = {}
        for index, feature in enumerate(FEATURES):
            with columns[index % 3]:
                minimum = float(ranges.loc["min", feature])
                maximum = float(ranges.loc["max", feature])
                center = float(medians[feature])
                span = max(maximum - minimum, abs(center) * 0.1, 1.0)
                entered[feature] = st.number_input(
                    FEATURE_LABELS[feature],
                    value=center,
                    min_value=float(minimum - span * 2),
                    max_value=float(maximum + span * 2),
                    format="%.5f" if feature == "rainfall_cv" else "%.3f",
                    help=f"Observed dataset range: {minimum:.4g} to {maximum:.4g}",
                )
        submitted = st.form_submit_button("Find matching catchment pattern", type="primary")

    if not submitted:
        st.info("Enter the values, then select **Find matching catchment pattern**.")
        return

    query = pd.DataFrame([entered], columns=FEATURES)
    if not np.isfinite(query.to_numpy(dtype=float)).all():
        st.error("Please provide finite numeric values for every feature.")
        return

    query_scaled = scaler.transform(query)
    predicted_cluster = int(model.predict(query_scaled)[0])
    cluster_distance = float(model.transform(query_scaled)[0, predicted_cluster])

    differences = x_scaled - query_scaled[0]
    distances = np.sqrt(np.sum(differences**2, axis=1))
    nearest_indices = np.argsort(distances)[:3]
    nearest = data.iloc[nearest_indices].copy()
    nearest["standardized_distance"] = distances[nearest_indices]
    nearest["cluster"] = labels[nearest_indices].astype(int)
    nearest["catchment_id"] = nearest["catchment_id"].astype(str)

    out_of_range = []
    for feature in FEATURES:
        low, high = float(ranges.loc["min", feature]), float(ranges.loc["max", feature])
        value = float(entered[feature])
        if value < low or value > high:
            out_of_range.append((feature, value, low, high))

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Assigned pattern", f"Cluster {predicted_cluster}")
    metric2.metric("Distance to cluster centroid", f"{cluster_distance:.3f}")
    metric3.metric("Closest reference catchment", nearest.iloc[0]["catchment_id"])

    if out_of_range:
        st.warning(
            "One or more entered values fall outside the observed training-data range. "
            "The result is an extrapolation and should be treated cautiously."
        )
        range_table = pd.DataFrame(
            [
                {
                    "Feature": FEATURE_LABELS[f],
                    "Entered value": value,
                    "Observed minimum": low,
                    "Observed maximum": high,
                }
                for f, value, low, high in out_of_range
            ]
        )
        st.dataframe(range_table, use_container_width=True, hide_index=True)

    st.markdown("#### Three closest catchments")
    display_nearest = nearest[
        ["catchment_id", "cluster", "standardized_distance", *FEATURES]
    ].rename(columns=FEATURE_LABELS)
    st.dataframe(display_nearest, use_container_width=True, hide_index=True)

    st.markdown("#### Entered values vs. closest catchment")
    comparison = pd.DataFrame(
        {
            "Feature": [FEATURE_LABELS[f] for f in FEATURES],
            "Entered catchment": [float(entered[f]) for f in FEATURES],
            "Closest reference": [float(nearest.iloc[0][f]) for f in FEATURES],
        }
    )
    comparison_long = comparison.melt(
        id_vars="Feature", var_name="Series", value_name="Value"
    )
    fig = px.bar(
        comparison_long,
        x="Feature",
        y="Value",
        color="Series",
        barmode="group",
        title="Feature-by-feature comparison (original units)",
    )
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Value (units differ by feature)",
        xaxis_tickangle=-35,
        legend_title="",
        margin={"l": 20, "r": 20, "t": 55, "b": 120},
    )
    st.plotly_chart(fig, use_container_width=True)

    cluster_members = data.loc[labels == predicted_cluster, FEATURES]
    cluster_profile = cluster_members.mean()
    standardized_profile = (cluster_profile - scaler.mean_) / scaler.scale_
    high_features = standardized_profile.sort_values(ascending=False).head(3)
    low_features = standardized_profile.sort_values().head(3)

    st.markdown(f"#### Cluster {predicted_cluster} profile")
    st.caption(
        f"This cluster contains {int((labels == predicted_cluster).sum())} reference "
        "catchments. The bullets below describe its average features relative to "
        "the full dataset average; they are not flood-risk labels."
    )
    st.write(
        "**Relatively higher features:** "
        + ", ".join(FEATURE_LABELS[f] for f in high_features.index)
    )
    st.write(
        "**Relatively lower features:** "
        + ", ".join(FEATURE_LABELS[f] for f in low_features.index)
    )

    st.markdown("#### Cluster mean profile (original units)")
    profile_frame = pd.DataFrame(
        {
            "Feature": [FEATURE_LABELS[f] for f in FEATURES],
            "Cluster mean": [float(cluster_profile[f]) for f in FEATURES],
            "Entered value": [float(entered[f]) for f in FEATURES],
        }
    )
    st.dataframe(profile_frame, use_container_width=True, hide_index=True)

    st.caption(
        "Distance is measured in standardized feature space. It is a similarity "
        "measure, not a calibrated confidence score. A nearest cluster assignment "
        "does not establish that a catchment will experience flooding."
    )



def _render_catchment_comparison(data_dir: Path, data: pd.DataFrame, scaler, labels) -> None:
    st.write(
        "Compare two reference catchments side by side using their nine model "
        "features, standardized pattern distance, cluster membership and historical monsoon rainfall."
    )
    panel_path = Path(data_dir) / "catchment_daily_rainfall_2019_2024.csv"
    if not panel_path.exists():
        st.error(f"Daily rainfall history was not found: {panel_path}")
        return
    try:
        panel = _load_daily_panel(str(panel_path))
    except Exception as exc:
        st.error(f"Could not load daily rainfall history: {exc}")
        return

    ids = data["catchment_id"].astype(str).tolist()
    if len(ids) < 2:
        st.info("At least two catchments are required for comparison.")
        return
    cluster_lookup = dict(zip(ids, labels.astype(int)))
    left, right = st.columns(2)
    with left:
        catchment_a = st.selectbox(
            "First catchment", ids, index=0,
            format_func=lambda cid: f"{cid} · Cluster {cluster_lookup[cid]}",
            key="compare_catchment_a",
        )
    with right:
        catchment_b = st.selectbox(
            "Second catchment", ids, index=1,
            format_func=lambda cid: f"{cid} · Cluster {cluster_lookup[cid]}",
            key="compare_catchment_b",
        )
    if catchment_a == catchment_b:
        st.info("Choose two different catchments to compare their profiles.")
        return

    row_a = data.loc[data["catchment_id"].astype(str) == catchment_a, FEATURES].iloc[0]
    row_b = data.loc[data["catchment_id"].astype(str) == catchment_b, FEATURES].iloc[0]
    pair_scaled = scaler.transform(pd.DataFrame([row_a, row_b], columns=FEATURES))
    distance = float(np.linalg.norm(pair_scaled[0] - pair_scaled[1]))

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{catchment_a} pattern", f"Cluster {cluster_lookup[catchment_a]}")
    c2.metric(f"{catchment_b} pattern", f"Cluster {cluster_lookup[catchment_b]}")
    c3.metric("Standardized feature distance", f"{distance:.3f}")

    compare_rows = []
    for feature in FEATURES:
        av = float(row_a[feature])
        bv = float(row_b[feature])
        compare_rows.append(
            {
                "Feature": FEATURE_LABELS[feature],
                catchment_a: av,
                catchment_b: bv,
                "Difference (B − A)": bv - av,
            }
        )
    comparison = pd.DataFrame(compare_rows)
    st.markdown("#### Feature-by-feature comparison")
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    st.download_button(
        "Download comparison (CSV)",
        data=comparison.to_csv(index=False).encode("utf-8"),
        file_name=f"catchment_comparison_{catchment_a}_{catchment_b}.csv",
        mime="text/csv",
    )

    z_values = scaler.transform(pd.DataFrame([row_a, row_b], columns=FEATURES))
    z_long = pd.DataFrame(z_values, columns=FEATURES)
    z_long["Catchment"] = [catchment_a, catchment_b]
    z_long = z_long.melt(id_vars="Catchment", var_name="Feature", value_name="Standardized value")
    z_long["Feature"] = z_long["Feature"].map(FEATURE_LABELS)
    fig = px.bar(
        z_long,
        x="Feature",
        y="Standardized value",
        color="Catchment",
        barmode="group",
        title="Relative feature profiles (standardized values)",
    )
    fig.update_layout(xaxis_tickangle=-35, xaxis_title="", legend_title="", margin={"b": 110})
    st.plotly_chart(fig, use_container_width=True)

    history = panel.loc[
        panel["catchment_id"].isin([catchment_a, catchment_b])
        & panel["date"].dt.month.isin([6, 7, 8, 9])
    ].copy()
    history["year"] = history["date"].dt.year
    annual = (
        history.groupby(["catchment_id", "year"], as_index=False)["rainfall_mm"]
        .sum()
        .rename(columns={"rainfall_mm": "monsoon_total_rainfall_mm"})
    )
    st.markdown("#### Historical monsoon rainfall (2019–2024)")
    fig_history = px.line(
        annual,
        x="year",
        y="monsoon_total_rainfall_mm",
        color="catchment_id",
        markers=True,
        title="Annual catchment-average monsoon rainfall total",
    )
    fig_history.update_layout(xaxis_title="Year", yaxis_title="Monsoon rainfall total (mm)", legend_title="Catchment")
    st.plotly_chart(fig_history, use_container_width=True)
    st.caption(
        "This is a descriptive comparison of historical rainfall and terrain. "
        "The standardized distance is not a flood-risk score or probability."
    )

def _render_rainfall_forecast(data_dir: Path, data: pd.DataFrame, labels) -> None:
    st.write(
        "Estimate the next day's catchment-area mean rainfall for one of the "
        "113 reference catchments using the preceding daily rainfall history."
    )
    st.warning(
        "Historical backtest mode only. The rainfall archive in this project ends "
        "on 31 December 2024, so this is not a live 2026 forecast or an operational warning."
    )

    panel_path = Path(data_dir) / "catchment_daily_rainfall_2019_2024.csv"
    project_root = Path(data_dir).resolve().parent.parent
    model_path = project_root / "models" / "next_day_rainfall_lgbm.joblib"
    if not panel_path.exists():
        st.error(
            f"Daily rainfall history was not found: {panel_path}. Add the generated "
            "daily panel under data/processed."
        )
        return
    if not model_path.exists():
        st.error(
            f"Forecast model was not found: {model_path}. Add the trained model "
            "under models/next_day_rainfall_lgbm.joblib."
        )
        return

    try:
        panel = _load_daily_panel(str(panel_path))
        artifact = _load_forecast_artifact(str(model_path))
    except Exception as exc:
        st.error(f"Could not load rainfall forecast inputs: {exc}")
        return

    panel_ids = sorted(panel["catchment_id"].unique().tolist())
    cluster_lookup = dict(zip(data["catchment_id"].astype(str), labels.astype(int)))
    options = [cid for cid in panel_ids if cid in cluster_lookup]
    if not options:
        st.error("No overlap was found between the daily rainfall panel and reference catchments.")
        return

    id_to_index = {str(cid): idx for idx, cid in enumerate(data["catchment_id"].astype(str))}
    selected_id = st.selectbox(
        "Select an existing catchment",
        options,
        format_func=lambda cid: f"{cid} · Cluster {cluster_lookup[cid]}",
        key="rainfall_forecast_catchment",
    )
    selected_date = st.date_input(
        "Historical forecast origin date",
        value=date(2024, 8, 15),
        min_value=date(2024, 6, 1),
        max_value=date(2024, 9, 29),
        help="The model uses rainfall observed up to this date to estimate the following day.",
        key="rainfall_forecast_origin_date",
    )

    origin = pd.Timestamp(selected_date)
    next_day = origin + pd.Timedelta(days=1)
    row_mask = (panel["catchment_id"] == selected_id) & (panel["date"] == origin)
    origin_rows = panel.loc[row_mask]
    actual_rows = panel.loc[
        (panel["catchment_id"] == selected_id) & (panel["date"] == next_day)
    ]
    if origin_rows.empty or actual_rows.empty:
        st.error("The selected catchment/date does not have a complete next-day record.")
        return

    origin_row = origin_rows.iloc[0]
    terrain_row = data.iloc[id_to_index[selected_id]]
    feature_values = origin_row.to_dict()
    feature_values["doy_sin"] = float(np.sin(2 * np.pi * origin.dayofyear / 365.25))
    feature_values["doy_cos"] = float(np.cos(2 * np.pi * origin.dayofyear / 365.25))
    for column in TERRAIN_FEATURES:
        feature_values[column] = float(terrain_row[column])
    input_frame = pd.DataFrame(
        [[feature_values[name] for name in FORECAST_FEATURES]],
        columns=FORECAST_FEATURES,
    )
    if not np.isfinite(input_frame.to_numpy(dtype=float)).all():
        st.error("The selected date does not have enough complete rainfall history for prediction.")
        return

    try:
        predicted_mm = float(artifact["model"].predict(input_frame)[0])
    except Exception as exc:
        st.error(f"Rainfall prediction failed: {exc}")
        return
    predicted_mm = max(0.0, predicted_mm)
    actual_mm = float(actual_rows.iloc[0]["rainfall_mm"])
    absolute_error = abs(predicted_mm - actual_mm)

    st.markdown(f"#### One-day-ahead estimate · {next_day.strftime('%d %B %Y')}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted catchment mean rainfall", f"{predicted_mm:.1f} mm")
    c2.metric("Observed next-day rainfall", f"{actual_mm:.1f} mm")
    c3.metric("Absolute forecast error", f"{absolute_error:.1f} mm")

    history = panel.loc[
        (panel["catchment_id"] == selected_id)
        & (panel["date"] >= origin - pd.Timedelta(days=6))
        & (panel["date"] <= origin),
        ["date", "rainfall_mm"],
    ].copy()
    history["series"] = "Observed history"
    next_points = pd.DataFrame(
        {
            "date": [next_day, next_day],
            "rainfall_mm": [predicted_mm, actual_mm],
            "series": ["Model estimate", "Observed next day"],
        }
    )
    chart_data = pd.concat([history, next_points], ignore_index=True)
    fig = px.line(
        chart_data,
        x="date",
        y="rainfall_mm",
        color="series",
        markers=True,
        title="Recent rainfall and next-day estimate",
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Catchment-area mean rainfall (mm/day)",
        legend_title="",
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
    )
    st.plotly_chart(fig, use_container_width=True)

    mae = float(artifact.get("walk_forward_mean_mae_mm", 7.2739))
    baseline_mae = float(artifact.get("walk_forward_persistence_mae_mm", 8.5724))
    improvement = 100 * (baseline_mae - mae) / baseline_mae
    st.markdown("#### Model validation")
    m1, m2, m3 = st.columns(3)
    m1.metric("Chronological holdout MAE", f"{mae:.2f} mm")
    m2.metric("Persistence baseline MAE", f"{baseline_mae:.2f} mm")
    m3.metric("MAE reduction vs baseline", f"{improvement:.1f}%")
    st.caption(
        "Validation used chronological annual holdouts (2022, 2023 and 2024). "
        "The model is trained on observations through 2023 and the 2024 monsoon "
        "dates are used here as a historical test window. The typical MAE is an "
        "aggregate metric, not a confidence interval for this individual estimate."
    )
    st.info(
        "This model estimates catchment-average rainfall from lagged observed rainfall "
        "and static terrain attributes. It does not use numerical weather forecasts, "
        "river discharge, soil moisture or observed flood labels. It must not be "
        "interpreted as flood probability, flood prediction or an early-warning signal."
    )


def render_pattern_explorer(data_dir: Path) -> None:
    """Render catchment pattern matching and a historical rainfall estimate tab."""
    st.subheader("Catchment Pattern Explorer")
    st.caption(
        "Explore similarity among the 113 modelled catchments and inspect a "
        "retrospective one-day-ahead rainfall estimation experiment."
    )

    dataset_path = Path(data_dir) / "clustering_dataset.csv"
    if not dataset_path.exists():
        st.error(f"Dataset not found: {dataset_path}")
        return

    try:
        data = _load_dataset(str(dataset_path))
        scaler, model, labels, x_scaled = _fit_reference_model(str(dataset_path))
    except Exception as exc:
        st.error(f"Could not prepare the reference model: {exc}")
        return

    st.caption(
        "Pattern matching uses all nine standardized rainfall and terrain features, "
        "with K-Means k=6 (seed 42, 20 initializations)."
    )
    tab_pattern, tab_compare, tab_forecast = st.tabs(
        ["Pattern matching", "Compare catchments", "24-hour rainfall estimate"]
    )
    with tab_pattern:
        _render_pattern_matching(data, scaler, model, labels, x_scaled)
    with tab_compare:
        _render_catchment_comparison(Path(data_dir), data, scaler, labels)
    with tab_forecast:
        _render_rainfall_forecast(Path(data_dir), data, labels)
