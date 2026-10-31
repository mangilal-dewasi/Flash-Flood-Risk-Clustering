from pathlib import Path
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.markdown(
    """
    <style>
    [data-testid="stToolbar"] {
        display: none !important;
    }

    [data-testid="stStatusWidget"] {
        display: none !important;
    }

    header[data-testid="stHeader"] {
    display: none !important;
    }
    
    .stAppDeployButton {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'processed'
FIGURES = ROOT / 'reports' / 'figures'

st.set_page_config(page_title='Flash-Flood Pattern Atlas', page_icon='🌧️', layout='wide', initial_sidebar_state='expanded')

st.markdown("""
<style>
[data-testid="stSidebar"] {
    min-width: 275px !important;
    max-width: 275px !important;
    width: 275px !important;
}

[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarCollapsedControl"] {
    display: none !important;
}

[data-testid="stSidebar"] > div:first-child {
    width: 275px !important;
}

/* Give the main dashboard more breathing room */
.block-container {
    padding-top: 1.5rem;
    max-width: 100%;
}
</style>
""", unsafe_allow_html=True)

# Premium research-dashboard styling: dark glass, restrained alpine teal, editorial typography.
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --line:rgba(173,203,222,.15); --text:#edf5fa; --muted:#9aafc2; --teal:#7ce0d2; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:radial-gradient(ellipse at 12% 0%,rgba(40,105,120,.22),transparent 34%),radial-gradient(ellipse at 92% 10%,rgba(63,86,145,.18),transparent 32%),linear-gradient(145deg,#07111f 0%,#0b1727 55%,#08121f 100%); color:var(--text); }
[data-testid="stHeader"] { background:rgba(7,17,31,.65); backdrop-filter:blur(18px); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,rgba(13,27,44,.96),rgba(9,20,34,.94)); border-right:1px solid var(--line); }
[data-testid="stSidebar"] > div:first-child { padding-top:1.4rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#b9c9d8; }
h1,h2,h3,h4 { font-family:'Manrope',sans-serif !important; letter-spacing:-.035em !important; color:#f2f7fb !important; }
h1 { font-weight:800 !important; } h2,h3 { font-weight:700 !important; } p,li,label { color:#c3d1df; }
.block-container { max-width:1500px; padding-top:2rem; padding-bottom:3rem; }
[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(29,51,73,.72),rgba(15,29,47,.72)); border:1px solid var(--line); border-radius:18px; padding:19px 20px; box-shadow:0 12px 35px rgba(0,0,0,.16),inset 0 1px 0 rgba(255,255,255,.045); backdrop-filter:blur(16px); }
[data-testid="stMetricLabel"] { color:#9eb4c7 !important; font-size:.78rem !important; text-transform:uppercase; letter-spacing:.09em; }
[data-testid="stMetricValue"] { color:#f3f8fc !important; font-family:'Manrope',sans-serif; font-weight:800; letter-spacing:-.04em; }
[data-testid="stAlert"] { border-radius:15px; border:1px solid rgba(124,224,210,.22); background:rgba(20,49,57,.55); color:#c9e8e5; }
[data-testid="stVerticalBlockBorderWrapper"] { border:1px solid var(--line) !important; border-radius:20px !important; background:linear-gradient(145deg,rgba(24,43,63,.68),rgba(13,27,44,.72)) !important; box-shadow:0 16px 45px rgba(0,0,0,.13),inset 0 1px 0 rgba(255,255,255,.035); backdrop-filter:blur(18px); }
[data-testid="stDataFrame"],[data-testid="stTable"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }
[data-testid="stPlotlyChart"] { background:linear-gradient(145deg,rgba(18,33,51,.58),rgba(12,25,41,.46)); border:1px solid var(--line); border-radius:18px; padding:8px 8px 2px; box-shadow:0 14px 38px rgba(0,0,0,.12); }
.stButton button,.stDownloadButton button { border-radius:12px; border:1px solid rgba(124,224,210,.32); background:linear-gradient(135deg,rgba(124,224,210,.18),rgba(108,164,220,.12)); color:#eafbf8; transition:all .2s ease; }
.stButton button:hover,.stDownloadButton button:hover { border-color:rgba(124,224,210,.72); transform:translateY(-1px); box-shadow:0 8px 24px rgba(42,150,145,.15); }
[data-baseweb="select"] > div,[data-baseweb="input"] > div { background:rgba(9,22,37,.8); border-color:var(--line); border-radius:11px; }
hr { border-color:var(--line); }
.atlas-eyebrow { color:var(--teal); font-family:'DM Mono',monospace; text-transform:uppercase; letter-spacing:.18em; font-size:.72rem; margin-bottom:.7rem; }
.atlas-hero { padding:1.8rem 2rem 1.65rem; border:1px solid rgba(163,203,225,.18); border-radius:24px; background:linear-gradient(120deg,rgba(23,48,66,.84),rgba(17,31,51,.70) 55%,rgba(26,39,67,.62)); box-shadow:0 22px 65px rgba(0,0,0,.2),inset 0 1px 0 rgba(255,255,255,.06); position:relative; overflow:hidden; margin-bottom:1.25rem; }
.atlas-hero:after { content:''; position:absolute; width:300px; height:300px; right:-70px; top:-155px; border-radius:50%; border:1px solid rgba(124,224,210,.17); box-shadow:0 0 0 26px rgba(124,224,210,.025),0 0 0 54px rgba(124,224,210,.02); }
.atlas-title { font-family:'Manrope',sans-serif; font-size:clamp(2rem,4vw,3.2rem); line-height:1.05; letter-spacing:-.06em; font-weight:800; color:#f3f8fc; margin:0 0 .75rem; }
.atlas-subtitle { max-width:850px; color:#b2c4d4; font-size:1rem; line-height:1.65; margin:0; }
.atlas-pill { display:inline-block; padding:.38rem .65rem; border:1px solid rgba(124,224,210,.25); border-radius:999px; color:#a9eee4; background:rgba(124,224,210,.07); font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; margin-bottom:1rem; }
@media(max-width:700px) { .block-container {padding:1rem 1rem 2rem;} .atlas-hero {padding:1.3rem;} }
</style>""", unsafe_allow_html=True)

FEATURE_LABELS = {
    'mean_monsoon_rainfall_mm': 'Mean monsoon rainfall (mm)',
    'mean_heavy_rain_days_50mm': 'Heavy-rain days ≥50 mm/year',
    'mean_very_heavy_rain_days_100mm': 'Very-heavy-rain days ≥100 mm/year',
    'mean_max_3day_rainfall_mm': 'Mean maximum 3-day rainfall (mm)',
    'rainfall_cv': 'Rainfall coefficient of variation',
    'mean_elevation_m': 'Mean elevation (m)',
    'relief_m': 'Relief (m)',
    'mean_slope_deg': 'Mean slope (degrees)',
    'slope_std_deg': 'Slope standard deviation (degrees)',
}

@st.cache_data
def load_csv(name):
    path = DATA / name
    if not path.exists():
        return None
    return pd.read_csv(path)

@st.cache_data
def load_catchment_shapes():
    path = DATA / 'final_catchments_L8_50km2.gpkg'
    if not path.exists():
        return None
    try:
        import geopandas as gpd
        gdf = gpd.read_file(path)
        # Streamlit/Plotly maps need WGS84 coordinates.
        if gdf.crs is not None and str(gdf.crs) != 'EPSG:4326':
            gdf = gdf.to_crs('EPSG:4326')
        return json.loads(gdf[['HYBAS_ID', 'geometry']].to_json())
    except Exception as exc:
        return {'error': str(exc)}

@st.cache_data
def pca_data(dataset):
    features = [c for c in dataset.columns if c != 'catchment_id']
    x = StandardScaler().fit_transform(dataset[features])
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(x)
    out = pd.DataFrame({'catchment_id': dataset['catchment_id'].astype(str), 'PC1': coords[:, 0], 'PC2': coords[:, 1]})
    return out, pca.explained_variance_ratio_, pca.components_, features

st.markdown("""<section class='atlas-hero'><div class='atlas-pill'>Himalayan catchment intelligence</div><h1 class='atlas-title'>Flash-Flood Pattern CLustering</h1><p class='atlas-subtitle'>An exploratory research interface for discovering rainfall–terrain patterns across hilly catchments in Himachal Pradesh and Uttarakhand. Built on IMD rainfall, SRTM-derived terrain features and HydroBASINS catchments.</p><div class='atlas-eyebrow'>Deisgned and Developed by Mangilal Dewasi</div></section>""", unsafe_allow_html=True)
st.warning('Research boundary · These clusters describe exploratory rainfall–terrain patterns. They are not validated flood-risk classes or flood-probability estimates; independent observed flood outcomes were not used.')

# Load the core data and handle missing optional analysis outputs gracefully.
dataset = load_csv('clustering_dataset.csv')
evaluation = load_csv('final_kmeans_evaluation.csv')
stability = load_csv('final_seed_stability_summary.csv')
method_comparison = load_csv('final_distance_method_comparison.csv')
hier_comparison = load_csv('final_hierarchical_comparison.csv')
if dataset is None:
    st.error('Missing data/processed/clustering_dataset.csv. Run this app from the project root and check that the processed dataset is present.')
    st.stop()

dataset['catchment_id'] = dataset['catchment_id'].astype(str)

k_choice = 6

with st.sidebar:
    st.header('Explore the analysis')

    st.markdown("""
    <div style="
        background: rgba(124,224,210,0.08);
        border: 1px solid rgba(124,224,210,0.25);
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 15px;
    ">
        <div style="font-size: 11px; color: #7ce0d2;">
            6 Clusters
        </div>
        <div style="font-size: 23px; font-weight: 700; color: white;">
            K-Means · K = 6
        </div>
    </div>
    """, unsafe_allow_html=True)

    view = st.radio(
        'Dashboard section',
        [
            'Overview',
            'Catchment map',
            'Cluster profiles',
            'PCA explorer',
            'Validation & methods',
            'Dataset explorer'
        ]
    )

    st.divider()
    st.caption(
        'Data: IMD daily rainfall (2019–2024), '
        'SRTM-derived terrain features, HydroBASINS L8 catchments.'
    )

assign_file = f'final_k{k_choice}_assignments.csv'
profiles_file = f'final_k{k_choice}_profiles_original_units.csv'
assignments = load_csv(assign_file)
profiles = load_csv(profiles_file)
if assignments is None:
    assignments = load_csv(f'kmeans_k{k_choice}_assignments.csv')
if assignments is None:
    st.error(f'Missing assignment file for k={k_choice}: {assign_file}')
    st.stop()
assignments['catchment_id'] = assignments['catchment_id'].astype(str)
joined = dataset.merge(assignments[['catchment_id', 'cluster'] + (['silhouette'] if 'silhouette' in assignments.columns else [])], on='catchment_id', how='left', validate='one_to_one')
joined['cluster_label'] = joined['cluster'].apply(lambda x: f'Cluster {int(x)}' if pd.notna(x) else 'Unassigned')

if view == 'Overview':
    st.subheader('Study snapshot')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Catchments modelled', f'{len(dataset):,}')
    c2.metric('Selected clusters', str(k_choice))
    c3.metric('Features', str(len(dataset.columns) - 1))
    metric_row = evaluation[evaluation['k'] == k_choice] if evaluation is not None and 'k' in evaluation.columns else pd.DataFrame()
    if not metric_row.empty:
        c4.metric('Silhouette score', f"{metric_row.iloc[0]['silhouette']:.4f}")
    else:
        c4.metric('Silhouette score', 'See report')
    st.markdown('### What the analysis does')
    st.write('Groups catchments using nine standardised rainfall and terrain features, then compares cluster counts and methods with internal validation metrics and random-seed stability.')
    st.markdown('### Cluster sizes')
    sizes = joined.groupby(['cluster', 'cluster_label'], as_index=False).size().rename(columns={'size': 'catchment_count'})
    fig = px.bar(sizes, x='cluster_label', y='catchment_count', color='cluster_label', text='catchment_count', title=f'Catchment count by K-Means cluster (k={k_choice})')
    fig.update_layout(showlegend=False, xaxis_title='', yaxis_title='Catchments')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'family':'DM Sans, sans-serif','color':'#d8e5ef'}, title={'font':{'family':'Manrope, sans-serif','size':18,'color':'#f1f7fb'}}, margin={'l':24,'r':24,'t':58,'b':28}, legend={'bgcolor':'rgba(0,0,0,0)','font':{'color':'#c5d4e0'}})
    fig.update_xaxes(gridcolor='rgba(160,190,210,.10)', zerolinecolor='rgba(160,190,210,.16)', linecolor='rgba(160,190,210,.15)')
    fig.update_yaxes(gridcolor='rgba(160,190,210,.10)', zerolinecolor='rgba(160,190,210,.16)', linecolor='rgba(160,190,210,.15)')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('### Interpretation guardrails')
    st.write('- Clusters describe similarity in the selected features, not measured flood occurrence.')
    st.write('- PCA is a two-dimensional view; clustering uses all nine standardised features.')
    st.write('- Small or weakly separated clusters should be interpreted cautiously.')

elif view == 'Catchment map':
    st.subheader(f'Catchment cluster map — K={k_choice}')
    geo = load_catchment_shapes()
    if geo is None:
        st.info('The catchment GeoPackage was not found. Add data/processed/final_catchments_L8_50km2.gpkg to enable the map.')
    elif isinstance(geo, dict) and 'error' in geo:
        st.error(f'Could not read catchment geometries: {geo["error"]}')
    else:
        # Geometry IDs in the GeoPackage are HydroBASINS IDs.
        for feature in geo.get('features', []):
            feature['id'] = str(feature['properties'].get('HYBAS_ID', ''))
        map_df = joined[['catchment_id', 'cluster_label', 'cluster']].copy()
        fig = px.choropleth_map(map_df, geojson=geo, locations='catchment_id', featureidkey='id', color='cluster_label',
                                map_style='carto-darkmatter', center={'lat': 31.5, 'lon': 78.2}, zoom=5.5,
                                opacity=0.72, hover_name='catchment_id', hover_data={'cluster_label': True, 'cluster': True, 'catchment_id': False},
                                title='Exploratory cluster membership by catchment')
        fig.update_layout(margin={'r': 0, 't': 45, 'l': 0, 'b': 0}, height=650)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'family':'DM Sans, sans-serif','color':'#d8e5ef'}, title={'font':{'family':'Manrope, sans-serif','size':18,'color':'#f1f7fb'}}, margin={'l':24,'r':24,'t':58,'b':28}, legend={'bgcolor':'rgba(0,0,0,0)','font':{'color':'#c5d4e0'}})
        fig.update_xaxes(gridcolor='rgba(160,190,210,.10)', zerolinecolor='rgba(160,190,210,.16)', linecolor='rgba(160,190,210,.15)')
        fig.update_yaxes(gridcolor='rgba(160,190,210,.10)', zerolinecolor='rgba(160,190,210,.16)', linecolor='rgba(160,190,210,.15)')
        st.plotly_chart(fig, use_container_width=True)
        st.caption('The map displays cluster membership, not flood probability. Catchments excluded from modelling may appear without a cluster.')

elif view == 'Cluster profiles':
    st.subheader(f'Cluster profiles — K={k_choice}')
    if profiles is None:
        st.warning(f'Profile file {profiles_file} is missing. Profiles are computed from the assigned catchments below.')
        profiles = joined.groupby('cluster', as_index=False)[list(FEATURE_LABELS)].mean()
        counts = joined.groupby('cluster').size().rename('catchment_count')
        profiles = profiles.merge(counts, left_on='cluster', right_index=True)
    if 'catchment_count' in profiles.columns:
        st.dataframe(profiles[['cluster', 'catchment_count'] + [c for c in FEATURE_LABELS if c in profiles.columns]].rename(columns=FEATURE_LABELS), use_container_width=True, hide_index=True)
    selected_feature = st.selectbox('Feature to compare across clusters', [c for c in FEATURE_LABELS if c in profiles.columns], format_func=lambda c: FEATURE_LABELS[c])
    fig = px.bar(profiles, x='cluster', y=selected_feature, color='cluster', text_auto='.3g', title=FEATURE_LABELS[selected_feature])
    fig.update_layout(showlegend=False, xaxis_title='Cluster', yaxis_title=FEATURE_LABELS[selected_feature])
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'family':'DM Sans, sans-serif','color':'#d8e5ef'}, title={'font':{'family':'Manrope, sans-serif','size':18,'color':'#f1f7fb'}}, margin={'l':24,'r':24,'t':58,'b':28}, legend={'bgcolor':'rgba(0,0,0,0)','font':{'color':'#c5d4e0'}})
    fig.update_xaxes(gridcolor='rgba(160,190,210,.10)', zerolinecolor='rgba(160,190,210,.16)', linecolor='rgba(160,190,210,.15)')
    fig.update_yaxes(gridcolor='rgba(160,190,210,.10)', zerolinecolor='rgba(160,190,210,.16)', linecolor='rgba(160,190,210,.15)')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('#### Feature profiles (standardised z-scores)')
    zprofiles = load_csv(f'final_k{k_choice}_profiles_zscores.csv')
    if zprofiles is not None:
        zcols = [c for c in FEATURE_LABELS if c in zprofiles.columns]
        zfig = go.Figure(data=go.Heatmap(z=zprofiles[zcols].to_numpy(), x=[FEATURE_LABELS[c] for c in zcols], y=[f"Cluster {int(c)}" for c in zprofiles['cluster']], colorscale='RdBu', zmid=0, colorbar_title='z-score'))
        zfig.update_layout(height=390, title='Above/below dataset average by cluster')
        st.plotly_chart(zfig, use_container_width=True)

elif view == 'PCA explorer':
    st.subheader('PCA projection')
    coords, ratios, components, features = pca_data(dataset)
    pca_joined = coords.merge(joined[['catchment_id', 'cluster_label', 'cluster'] + (['silhouette'] if 'silhouette' in joined.columns else [])], on='catchment_id', how='left')
    fig = px.scatter(pca_joined, x='PC1', y='PC2', color='cluster_label', hover_name='catchment_id',
                     hover_data=['cluster'] + (['silhouette'] if 'silhouette' in pca_joined.columns else []),
                     title=f'PCA view coloured by K-Means clusters (k={k_choice})')
    fig.update_layout(xaxis_title=f'PC1 ({ratios[0]*100:.2f}% variance)', yaxis_title=f'PC2 ({ratios[1]*100:.2f}% variance)')
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'family':'DM Sans, sans-serif','color':'#d8e5ef'}, title={'font':{'family':'Manrope, sans-serif','size':18,'color':'#f1f7fb'}}, margin={'l':24,'r':24,'t':58,'b':28}, legend={'bgcolor':'rgba(0,0,0,0)','font':{'color':'#c5d4e0'}})
    fig.update_xaxes(gridcolor='rgba(160,190,210,.10)', zerolinecolor='rgba(160,190,210,.16)', linecolor='rgba(160,190,210,.15)')
    fig.update_yaxes(gridcolor='rgba(160,190,210,.10)', zerolinecolor='rgba(160,190,210,.16)', linecolor='rgba(160,190,210,.15)')
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f'PC1 + PC2 explain {ratios.sum()*100:.2f}% of variance. Clustering itself uses all nine standardised features.')
    loadings = pd.DataFrame(components.T, index=features, columns=['PC1 loading', 'PC2 loading']).rename(index=FEATURE_LABELS)
    st.markdown('#### PCA component loadings')
    st.dataframe(loadings, use_container_width=True)

elif view == 'Validation & methods':
    st.subheader('Validation and method comparison')
    if evaluation is not None:
        st.markdown('#### K-Means sweep (Euclidean)')
        st.dataframe(evaluation, use_container_width=True, hide_index=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=evaluation['k'], y=evaluation['silhouette'], mode='lines+markers', name='Silhouette (higher is better)'))
        fig.add_trace(go.Scatter(x=evaluation['k'], y=evaluation['davies_bouldin'], mode='lines+markers', name='Davies–Bouldin (lower is better)', yaxis='y2'))
        fig.update_layout(title='K-Means validation across k', xaxis_title='Number of clusters (k)', yaxis={'title': 'Silhouette'}, yaxis2={'title': 'Davies–Bouldin', 'overlaying': 'y', 'side': 'right'})
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'family':'DM Sans, sans-serif','color':'#d8e5ef'}, title={'font':{'family':'Manrope, sans-serif','size':18,'color':'#f1f7fb'}}, margin={'l':24,'r':24,'t':58,'b':28}, legend={'bgcolor':'rgba(0,0,0,0)','font':{'color':'#c5d4e0'}})
        fig.update_xaxes(gridcolor='rgba(160,190,210,.10)', zerolinecolor='rgba(160,190,210,.16)', linecolor='rgba(160,190,210,.15)')
        fig.update_yaxes(gridcolor='rgba(160,190,210,.10)', zerolinecolor='rgba(160,190,210,.16)', linecolor='rgba(160,190,210,.15)')
        st.plotly_chart(fig, use_container_width=True)
    if stability is not None:
        st.markdown('#### Random-seed stability')
        st.dataframe(stability, use_container_width=True, hide_index=True)
    if method_comparison is not None:
        st.markdown('#### Euclidean K-Means vs Manhattan K-Medoids')
        st.dataframe(method_comparison, use_container_width=True, hide_index=True)
    else:
        st.info('Distance comparison file not present in the project yet. Expected: data/processed/final_distance_method_comparison.csv')
    if hier_comparison is not None:
        st.markdown('#### Hierarchical clustering comparison')
        st.dataframe(hier_comparison, use_container_width=True, hide_index=True)
    else:
        st.info('Hierarchical comparison file not present in the project yet. Expected: data/processed/final_hierarchical_comparison.csv')
    st.warning('Metrics from different distance definitions are not always directly comparable. Manhattan silhouette should use Manhattan distances; Euclidean Davies–Bouldin/Calinski–Harabasz values are supplementary diagnostics.')

elif view == 'Dataset explorer':
    st.subheader('Catchment-level dataset')
    feature_options = [c for c in dataset.columns if c != 'catchment_id']
    selected_features = st.multiselect('Features to display', feature_options, default=feature_options[:5], format_func=lambda c: FEATURE_LABELS.get(c, c))
    filter_cluster = st.multiselect('Filter clusters', sorted(joined['cluster'].dropna().unique().astype(int).tolist()), default=sorted(joined['cluster'].dropna().unique().astype(int).tolist()))
    shown = joined[joined['cluster'].isin(filter_cluster)][['catchment_id', 'cluster'] + selected_features]
    st.dataframe(shown, use_container_width=True, hide_index=True)
    st.download_button('Download filtered catchments (CSV)', data=shown.to_csv(index=False).encode('utf-8'), file_name=f'catchment_clusters_k{k_choice}.csv', mime='text/csv')
    st.markdown('#### Feature distributions')
    dist_feature = st.selectbox('Distribution feature', feature_options, format_func=lambda c: FEATURE_LABELS.get(c, c))
    fig = px.histogram(joined, x=dist_feature, color='cluster_label', marginal='box', barmode='overlay', opacity=0.65, title=FEATURE_LABELS.get(dist_feature, dist_feature))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font={'family':'DM Sans, sans-serif','color':'#d8e5ef'}, title={'font':{'family':'Manrope, sans-serif','size':18,'color':'#f1f7fb'}}, margin={'l':24,'r':24,'t':58,'b':28}, legend={'bgcolor':'rgba(0,0,0,0)','font':{'color':'#c5d4e0'}})
    fig.update_xaxes(gridcolor='rgba(160,190,210,.10)', zerolinecolor='rgba(160,190,210,.16)', linecolor='rgba(160,190,210,.15)')
    fig.update_yaxes(gridcolor='rgba(160,190,210,.10)', zerolinecolor='rgba(160,190,210,.16)', linecolor='rgba(160,190,210,.15)')
    st.plotly_chart(fig, use_container_width=True)

st.divider()