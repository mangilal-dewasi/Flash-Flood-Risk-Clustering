"""Rebuild the Flash-Flood Pattern Atlas processed data and analysis outputs.

Run from the repository root after downloading the source datasets:

    python download.py
    python pipeline.py
    streamlit run app.py

The pipeline deliberately runs only the scripts required for the dashboard's
final data products. Older/provisional analysis scripts are not run because
some write overlapping CSV filenames and could overwrite the consolidated
results.

Expected project layout:
    download.py
    app.py
    pipeline.py
    src/
        extract_rainfall_features.py
        extract_terrain_features.py
        build_modeling_dataset.py
        select_clustering_features.py
        final_project_analysis_pca_manhattan.py
    data/
        interim/western_himalaya_study_boundary.shp (+ sidecars)
        raw/hydrobasins/...
        raw/imd/RF25_2019.nc ... RF25_2024.nc
        raw/srtm/*.hgt
        interim/
        processed/

All generated CSV files are written by the source scripts to
`data/processed/`. The combined rainfall NetCDF is written to `data/interim/`.
The final catchment GeoPackage is written to `data/processed/` for app.py.
"""
from __future__ import annotations

import argparse
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
SRC = ROOT / "src"

STUDY_BOUNDARY = INTERIM / "western_himalaya_study_boundary.shp"
CANDIDATE_CATCHMENTS = INTERIM / "western_himalaya_catchments_L8.gpkg"
FINAL_CATCHMENTS = PROCESSED / "final_catchments_L8_50km2.gpkg"
COMBINED_RAINFALL = INTERIM / "imd_rainfall_2019_2024_combined.nc"

YEARS = range(2019, 2025)
MIN_OVERLAP_PERCENT = 50.0
MIN_CATCHMENT_AREA_KM2 = 50.0
EQUAL_AREA_CRS = "EPSG:6933"

REQUIRED_SCRIPTS = [
    SRC / "extract_rainfall_features.py",
    SRC / "extract_terrain_features.py",
    SRC / "build_modeling_dataset.py",
    SRC / "select_clustering_features.py",
    # This consolidated script creates the final Euclidean/Manhattan
    # comparison, PCA outputs, final assignments, profiles and metrics.
    SRC / "final_project_analysis_pca_manhattan.py",
]

EXPECTED_FINAL_CSVS = [
    "catchment_rainfall_features.csv",
    "catchment_terrain_features.csv",
    "catchment_features_merged_all.csv",
    "rainfall_coverage_exclusions.csv",
    "modeling_dataset.csv",
    "feature_summary.csv",
    "feature_correlation.csv",
    "clustering_dataset.csv",
    "clustering_feature_correlation.csv",
    "final_feature_summary.csv",
    "final_feature_correlation.csv",
    "final_high_feature_correlations.csv",
    "final_kmeans_evaluation.csv",
    "final_seed_stability_summary.csv",
    "final_seed_metrics.csv",
    "final_k5_assignments.csv",
    "final_k6_assignments.csv",
    "final_k5_profiles_original_units.csv",
    "final_k6_profiles_original_units.csv",
    "final_k5_profiles_zscores.csv",
    "final_k6_profiles_zscores.csv",
    "final_k5_silhouette_by_cluster.csv",
    "final_k6_silhouette_by_cluster.csv",
    "final_cluster_profiles_both_k.csv",
    "final_cluster_silhouette_summaries.csv",
    "final_assignments_k5_k6.csv",
    "final_distance_method_comparison.csv",
    "final_distance_method_assignments.csv",
    "final_pca_explained_variance.csv",
    "final_pca_loadings.csv",
]


def print_stage(number: int, title: str) -> None:
    print("\n" + "=" * 78, flush=True)
    print(f"STAGE {number}: {title}", flush=True)
    print("=" * 78, flush=True)


def require_file(path: Path, purpose: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {purpose}: {path.relative_to(ROOT)}")
    if path.stat().st_size == 0:
        raise RuntimeError(f"File is empty ({purpose}): {path.relative_to(ROOT)}")


def check_shapefile_sidecars(shp_path: Path, label: str) -> None:
    require_file(shp_path, f"{label} shapefile")
    for extension in (".shx", ".dbf", ".prj"):
        sidecar = shp_path.with_suffix(extension)
        require_file(sidecar, f"{label} shapefile sidecar")


def find_hydrobasins_shapefile() -> Path:
    """Locate the required HydroBASINS Level 8 Asia shapefile recursively."""
    search_roots = [RAW / "hydrobasins", DATA / "hydrobasins"]
    matches: list[Path] = []
    for search_root in search_roots:
        if search_root.exists():
            matches.extend(
                p for p in search_root.rglob("*.shp")
                if p.name.lower() == "hybas_as_lev08_v1c.shp"
            )
    matches = sorted(set(matches))

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            "More than one hybas_as_lev08_v1c.shp was found. Keep one unambiguous "
            "HydroBASINS source under data/raw/hydrobasins/."
        )

    # A single shapefile in the HydroBASINS folder can be accepted when the
    # archive uses a different filename, but ambiguity is never guessed.
    candidates: list[Path] = []
    for search_root in search_roots:
        if search_root.exists():
            candidates.extend(search_root.rglob("*.shp"))
    candidates = sorted(set(candidates))
    if len(candidates) == 1:
        print(
            "Warning: HydroBASINS shapefile has a non-standard name: "
            f"{candidates[0].name}", flush=True
        )
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            "No HydroBASINS shapefile found. Run `python download.py` first. "
            "Expected hybas_as_lev08_v1c.shp under data/raw/hydrobasins/."
        )
    raise RuntimeError(
        "Could not identify the HydroBASINS Level 8 shapefile because multiple "
        "shapefiles were found: " + ", ".join(p.name for p in candidates)
    )


def locate_annual_rainfall_file(year: int) -> Path:
    """Support the new download.py location and the older local layout."""
    candidates = [
        RAW / "imd" / f"RF25_{year}.nc",
        DATA / f"RF25_{year}.nc",
    ]
    for path in candidates:
        if path.is_file() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(
        f"Missing IMD annual rainfall file for {year}. Run `python download.py`; "
        f"expected data/raw/imd/RF25_{year}.nc."
    )


def combine_annual_rainfall() -> None:
    """Combine RF25_2019.nc ... RF25_2024.nc into the feature script input."""
    try:
        import xarray as xr
    except ImportError as exc:
        raise RuntimeError(
            "xarray is required to combine the IMD NetCDF files. "
            "Install the full project requirements first."
        ) from exc

    sources: list[Path] = [locate_annual_rainfall_file(year) for year in YEARS]
    datasets = []
    combined = None
    temporary_output = COMBINED_RAINFALL.with_name(
        COMBINED_RAINFALL.stem + ".tmp.nc"
    )

    try:
        for year, path in zip(YEARS, sources):
            print(f"Opening rainfall source: {path.relative_to(ROOT)}", flush=True)
            ds = xr.open_dataset(path)
            datasets.append(ds)

            if "RAINFALL" not in ds.data_vars:
                raise ValueError(f"{path.name} does not contain the RAINFALL variable.")
            if "TIME" not in ds.coords or "TIME" not in ds.dims:
                raise ValueError(f"{path.name} does not contain the expected TIME dimension.")
            if not {"LATITUDE", "LONGITUDE"}.issubset(ds.coords):
                raise ValueError(
                    f"{path.name} must contain LATITUDE and LONGITUDE coordinates."
                )
            if set(np.asarray(ds["TIME"].dt.year.values).tolist()) != {year}:
                raise ValueError(
                    f"{path.name} should contain only {year} observations."
                )
            if tuple(ds["RAINFALL"].dims) != ("TIME", "LATITUDE", "LONGITUDE"):
                raise ValueError(
                    f"Unexpected RAINFALL dimensions in {path.name}: "
                    f"{ds['RAINFALL'].dims}"
                )

        # All years must use the same grid; otherwise concat would silently
        # expand/reindex spatial coordinates and alter the analysis footprint.
        reference = datasets[0]
        for path, ds in zip(sources[1:], datasets[1:]):
            for coordinate in ("LATITUDE", "LONGITUDE"):
                if not np.array_equal(
                    reference[coordinate].values, ds[coordinate].values
                ):
                    raise ValueError(
                        f"Spatial grid mismatch in {path.name} ({coordinate}). "
                        "Do not combine annual files from different IMD grids."
                    )

        combined = xr.concat(
            datasets,
            dim="TIME",
            join="exact",
            data_vars="minimal",
            coords="minimal",
            compat="equals",
        ).sortby("TIME")

        time_values = pd.to_datetime(combined["TIME"].values)
        if time_values.has_duplicates:
            raise ValueError("Duplicate timestamps found in the combined rainfall files.")
        actual_years = set(time_values.year.tolist())
        if actual_years != set(YEARS):
            raise ValueError(
                f"Combined rainfall years are {sorted(actual_years)}, "
                f"expected {list(YEARS)}."
            )

        INTERIM.mkdir(parents=True, exist_ok=True)
        temporary_output.unlink(missing_ok=True)
        print(
            f"Writing combined rainfall NetCDF ({len(time_values)} time steps)...",
            flush=True,
        )
        combined.to_netcdf(temporary_output)
        combined.close()
        combined = None
        temporary_output.replace(COMBINED_RAINFALL)
        print(f"Created: {COMBINED_RAINFALL.relative_to(ROOT)}", flush=True)

    finally:
        if combined is not None:
            combined.close()
        for ds in datasets:
            ds.close()
        temporary_output.unlink(missing_ok=True)


def _repair_invalid_geometries(geometries):
    """Repair invalid polygon geometries when possible, without hiding failures."""
    invalid = ~geometries.is_valid
    invalid_count = int(invalid.sum())
    if not invalid_count:
        return geometries

    print(f"Repairing {invalid_count} invalid geometries with buffer(0)...", flush=True)
    repaired = geometries.copy()
    repaired.loc[invalid] = repaired.loc[invalid].buffer(0)
    still_invalid = ~repaired.is_valid
    if still_invalid.any():
        raise ValueError(
            f"{int(still_invalid.sum())} geometries remain invalid after repair."
        )
    return repaired


def prepare_final_catchments() -> None:
    """Build the study-area and >=50 km² catchment layers used downstream.

    This implements the same two filters documented in the project's earlier
    catchment scripts: >=50% catchment overlap with the study boundary, then
    >=50 km² catchment area. It writes both candidate and final GeoPackages.
    """
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError(
            "geopandas is required to prepare the catchment GeoPackage. "
            "Install the full project requirements first."
        ) from exc

    check_shapefile_sidecars(STUDY_BOUNDARY, "Study boundary")
    hydro_file = find_hydrobasins_shapefile()
    check_shapefile_sidecars(hydro_file, "HydroBASINS")

    print(f"Reading HydroBASINS: {hydro_file.relative_to(ROOT)}", flush=True)
    basins = gpd.read_file(hydro_file)
    print(f"Reading study boundary: {STUDY_BOUNDARY.relative_to(ROOT)}", flush=True)
    study = gpd.read_file(STUDY_BOUNDARY)

    if basins.empty or study.empty:
        raise ValueError("HydroBASINS or study-boundary shapefile contains no records.")
    if "HYBAS_ID" not in basins.columns:
        raise ValueError(
            "HydroBASINS layer is missing HYBAS_ID. Check that the Level 8 Asia layer was downloaded."
        )
    if basins.crs is None or study.crs is None:
        raise ValueError(
            "CRS metadata is missing from HydroBASINS or the study-boundary shapefile."
        )

    basins = basins.loc[basins.geometry.notna() & ~basins.geometry.is_empty].copy()
    study = study.loc[study.geometry.notna() & ~study.geometry.is_empty].copy()
    if basins.empty or study.empty:
        raise ValueError("No valid polygon geometries remain in the input shapefiles.")

    basins = basins.set_geometry(_repair_invalid_geometries(basins.geometry))
    study = study.set_geometry(_repair_invalid_geometries(study.geometry))

    # Calculate both areas and overlap in an equal-area projection.
    basins_equal = basins.to_crs(EQUAL_AREA_CRS)
    study_equal = study.to_crs(EQUAL_AREA_CRS)
    study_union = study_equal.geometry.unary_union

    basin_area_km2 = basins_equal.geometry.area / 1_000_000.0
    intersection_area_km2 = (
        basins_equal.geometry.intersection(study_union).area / 1_000_000.0
    )
    overlap_pct = np.where(
        basin_area_km2.to_numpy() > 0,
        intersection_area_km2.to_numpy() / basin_area_km2.to_numpy() * 100.0,
        0.0,
    )

    basins_equal["catchment_area_km2"] = basin_area_km2
    basins_equal["area_km2"] = basin_area_km2
    basins_equal["intersection_area_km2"] = intersection_area_km2
    basins_equal["study_overlap_pct"] = overlap_pct

    candidates = basins_equal.loc[
        basins_equal["study_overlap_pct"] >= MIN_OVERLAP_PERCENT
    ].copy()
    if candidates.empty:
        raise RuntimeError(
            "No HydroBASINS catchments met the 50% study-area overlap threshold. "
            "Check the boundary CRS and the HydroBASINS layer."
        )

    final = candidates.loc[
        candidates["catchment_area_km2"] >= MIN_CATCHMENT_AREA_KM2
    ].copy()
    if final.empty:
        raise RuntimeError(
            "No catchments met the 50 km² minimum-area threshold after overlap filtering."
        )

    INTERIM.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    # GeoPackage outputs must be replaced cleanly to avoid stale layers.
    for output in (CANDIDATE_CATCHMENTS, FINAL_CATCHMENTS):
        output.unlink(missing_ok=True)

    candidates.to_crs("EPSG:4326").to_file(
        CANDIDATE_CATCHMENTS,
        layer="western_himalaya_catchments_L8",
        driver="GPKG",
    )
    final.to_crs("EPSG:4326").to_file(
        FINAL_CATCHMENTS,
        layer="final_catchments",
        driver="GPKG",
    )

    print(f"HydroBASINS source catchments: {len(basins_equal)}", flush=True)
    print(f"Catchments with >=50% study overlap: {len(candidates)}", flush=True)
    print(f"Final catchments with >=50 km² area: {len(final)}", flush=True)
    print(f"Created: {CANDIDATE_CATCHMENTS.relative_to(ROOT)}", flush=True)
    print(f"Created: {FINAL_CATCHMENTS.relative_to(ROOT)}", flush=True)


def run_script(script: Path, label: str) -> None:
    if not script.is_file():
        raise FileNotFoundError(
            f"Required source script not found: {script.relative_to(ROOT)}"
        )
    print(f"\nRunning {script.relative_to(ROOT)} ...", flush=True)
    # Execute with the current Python interpreter and repository root as CWD.
    # Existing scripts use Path(__file__).parents[1], so their outputs resolve
    # to this repository's data/processed and reports/ directories.
    subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        check=True,
    )
    print(f"Completed: {label}", flush=True)


def verify_csv_outputs() -> None:
    """Check required files and schemas consumed by app.py / Objective 2 UI."""
    missing = [name for name in EXPECTED_FINAL_CSVS if not (PROCESSED / name).is_file()]
    empty = [
        name for name in EXPECTED_FINAL_CSVS
        if (PROCESSED / name).is_file() and (PROCESSED / name).stat().st_size == 0
    ]
    if missing or empty:
        messages = []
        if missing:
            messages.append("Missing outputs: " + ", ".join(missing))
        if empty:
            messages.append("Empty outputs: " + ", ".join(empty))
        raise RuntimeError("Pipeline output validation failed. " + " | ".join(messages))

    required_columns = {
        "clustering_dataset.csv": {"catchment_id"},
        "final_kmeans_evaluation.csv": {
            "k", "inertia", "silhouette", "davies_bouldin", "smallest_cluster_size"
        },
        "final_k5_assignments.csv": {"catchment_id", "cluster"},
        "final_k6_assignments.csv": {"catchment_id", "cluster"},
        "final_k5_profiles_original_units.csv": {"catchment_count"},
        "final_k6_profiles_original_units.csv": {"catchment_count"},
        "final_distance_method_comparison.csv": {
            "k", "method", "distance", "silhouette_matching_distance"
        },
    }
    for filename, expected in required_columns.items():
        df = pd.read_csv(PROCESSED / filename, nrows=5)
        absent = expected - set(df.columns)
        if absent:
            raise RuntimeError(
                f"{filename} is missing required columns: {sorted(absent)}"
            )

    data = pd.read_csv(PROCESSED / "clustering_dataset.csv", dtype={"catchment_id": str})
    if data.empty:
        raise RuntimeError("clustering_dataset.csv contains no catchments.")
    if data["catchment_id"].isna().any() or data["catchment_id"].duplicated().any():
        raise RuntimeError("clustering_dataset.csv has missing or duplicate catchment IDs.")
    numeric = data.drop(columns=["catchment_id"]).apply(pd.to_numeric, errors="coerce")
    if numeric.empty or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise RuntimeError(
            "clustering_dataset.csv contains non-numeric, missing, or infinite feature values."
        )

    expected_rows = len(data)
    for filename in ("final_k5_assignments.csv", "final_k6_assignments.csv"):
        assignments = pd.read_csv(
            PROCESSED / filename, dtype={"catchment_id": str}
        )
        if len(assignments) != expected_rows:
            raise RuntimeError(
                f"{filename} has {len(assignments)} rows; expected {expected_rows} "
                "to match clustering_dataset.csv."
            )
        if assignments["catchment_id"].duplicated().any():
            raise RuntimeError(f"{filename} contains duplicate catchment IDs.")

    print("\nValidated dashboard-critical CSVs:", flush=True)
    for name in EXPECTED_FINAL_CSVS:
        file_path = PROCESSED / name
        print(f"  ✓ {name} ({file_path.stat().st_size:,} bytes)", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild all processed datasets and final clustering results."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check whether the required downloaded inputs and source scripts exist.",
    )
    return parser.parse_args()


def preflight() -> None:
    """Check required downloaded datasets and versioned source scripts."""
    missing_scripts = [p.relative_to(ROOT) for p in REQUIRED_SCRIPTS if not p.is_file()]
    if missing_scripts:
        raise FileNotFoundError(
            "Missing required source scripts: "
            + ", ".join(str(p) for p in missing_scripts)
            + ". Keep these scripts in the repository's src/ folder."
        )

    for year in YEARS:
        locate_annual_rainfall_file(year)

    hgt_files = sorted((RAW / "srtm").rglob("*.hgt")) if (RAW / "srtm").exists() else []
    if len(hgt_files) != 42:
        raise RuntimeError(
            "The current extract_terrain_features.py expects exactly 42 SRTM .hgt tiles, "
            f"but found {len(hgt_files)} in data/raw/srtm/. Run download.py and verify "
            "that the SRTM ZIP contains the complete tile set."
        )

    check_shapefile_sidecars(STUDY_BOUNDARY, "Study boundary")
    hydro_file = find_hydrobasins_shapefile()
    check_shapefile_sidecars(hydro_file, "HydroBASINS")

    print("Preflight checks passed.", flush=True)
    print(f"HydroBASINS: {hydro_file.relative_to(ROOT)}", flush=True)
    print(f"Study boundary: {STUDY_BOUNDARY.relative_to(ROOT)}", flush=True)
    print(f"IMD files: {len(list(YEARS))} annual NetCDF files (2019–2024)", flush=True)
    print(f"SRTM tiles: {len(hgt_files)} HGT files", flush=True)


def main() -> int:
    args = parse_args()
    print("FLASH-FLOOD PATTERN ATLAS — REPRODUCIBLE PIPELINE", flush=True)
    print(f"Repository root: {ROOT}", flush=True)
    print("CSV output directory: data/processed/", flush=True)

    for folder in (RAW / "hydrobasins", RAW / "imd", RAW / "srtm", INTERIM, PROCESSED, REPORTS, FIGURES):
        folder.mkdir(parents=True, exist_ok=True)

    try:
        print_stage(0, "Preflight checks")
        preflight()
        if args.check_only:
            print("\nCheck-only complete. Inputs and source scripts are present.", flush=True)
            return 0

        print_stage(1, "Combine IMD annual rainfall NetCDF files")
        combine_annual_rainfall()
        require_file(COMBINED_RAINFALL, "combined IMD rainfall NetCDF")

        print_stage(2, "Prepare final HydroBASINS catchments")
        prepare_final_catchments()
        require_file(FINAL_CATCHMENTS, "final catchment GeoPackage")

        print_stage(3, "Extract catchment-level rainfall features")
        run_script(SRC / "extract_rainfall_features.py", "rainfall feature extraction")
        require_file(PROCESSED / "catchment_rainfall_features.csv", "rainfall feature CSV")

        print_stage(4, "Extract catchment-level terrain features")
        run_script(SRC / "extract_terrain_features.py", "terrain feature extraction")
        require_file(PROCESSED / "catchment_terrain_features.csv", "terrain feature CSV")

        print_stage(5, "Merge, validate and build the modeling dataset")
        run_script(SRC / "build_modeling_dataset.py", "modeling dataset preparation")
        require_file(PROCESSED / "modeling_dataset.csv", "modeling dataset CSV")

        print_stage(6, "Select the nine clustering features")
        run_script(SRC / "select_clustering_features.py", "clustering feature selection")
        require_file(PROCESSED / "clustering_dataset.csv", "clustering dataset CSV")

        print_stage(7, "Run consolidated clustering, distance comparison and PCA analysis")
        run_script(
            SRC / "final_project_analysis_pca_manhattan.py",
            "final Euclidean K-Means / Manhattan K-Medoids / PCA analysis",
        )

        print_stage(8, "Verify files consumed by the dashboard")
        verify_csv_outputs()
        require_file(REPORTS / "final_analysis_report.txt", "final analysis report")
        require_file(FINAL_CATCHMENTS, "catchment GeoPackage used by the map")

    except subprocess.CalledProcessError as exc:
        print(
            f"\nPIPELINE FAILED: {exc.cmd} exited with code {exc.returncode}. "
            "Fix the first error shown above; later stages were not run.",
            file=sys.stderr,
            flush=True,
        )
        return int(exc.returncode or 1)
    except Exception as exc:
        print(f"\nPIPELINE FAILED: {exc}", file=sys.stderr, flush=True)
        return 1

    print("\n" + "=" * 78, flush=True)
    print("PIPELINE COMPLETED SUCCESSFULLY", flush=True)
    print("Dashboard data: data/processed/", flush=True)
    print("Reports/figures: reports/", flush=True)
    print("Next command: streamlit run app.py", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
