from pathlib import Path
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import geometry_mask


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRTM_DIR = PROJECT_ROOT / "data" / "raw" / "srtm"
CATCHMENT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_catchments_L8_50km2.gpkg"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "catchment_terrain_features.csv"


# ============================================================
# CONFIGURATION
# ============================================================

# SRTM HGT files are big-endian signed 16-bit integers.
SRTM_DTYPE = ">i2"

# SRTM NoData value
NODATA_VALUE = -32768

# Maximum number of pixels used for exact slope statistics
# If a catchment is very large, processing remains manageable.
MAX_SLOPE_PIXELS = 2_000_000


# ============================================================
# HGT READER
# ============================================================

def read_hgt(path):
    """
    Read one SRTM HGT tile.

    A standard 1-degree SRTM tile at 1 arc-second resolution
    contains 3601 x 3601 signed 16-bit elevation samples.
    """

    data = np.fromfile(path, dtype=SRTM_DTYPE)

    expected = 3601 * 3601

    if data.size != expected:
        raise ValueError(
            f"Unexpected HGT size for {path.name}: "
            f"{data.size} samples, expected {expected}"
        )

    data = data.reshape((3601, 3601)).astype(np.float32)

    data[data == NODATA_VALUE] = np.nan

    # Tile naming: N28E075, SXXWXXX etc.
    name = path.stem

    lat_sign = 1 if name[0] == "N" else -1
    lon_start = 1 if "E" in name else -1

    lat_deg = int(name[1:3]) * lat_sign

    e_index = name.find("E")
    w_index = name.find("W")

    if e_index != -1:
        lon_deg = int(name[e_index + 1 :]) * 1
    else:
        lon_deg = -int(name[w_index + 1 :])

    # HGT begins at the NW corner.
    transform = from_origin(
        lon_deg,
        lat_deg + 1,
        1 / 3600,
        1 / 3600,
    )

    return data, transform


# ============================================================
# BUILD SRTM MOSAIC
# ============================================================

def build_mosaic():
    """
    Build a georeferenced SRTM mosaic from all extracted HGT tiles.

    Rasterio/GDAL handles the geographic placement of each tile,
    avoiding manual row/column indexing errors at tile boundaries.
    """

    hgt_files = sorted(SRTM_DIR.glob("*.hgt"))

    if len(hgt_files) != 42:
        raise RuntimeError(
            f"Expected 42 HGT tiles, found {len(hgt_files)}"
        )

    datasets = []

    try:
        for path in hgt_files:
            print(f"Reading {path.name}")

            # HGT files are standard SRTM rasters.
            src = rasterio.open(
                path,
                driver="SRTMHGT",
            )

            datasets.append(src)

        print("\nMerging SRTM tiles...")

        mosaic, transform = merge(
            datasets,
            nodata=np.nan,
            dtype="float32",
        )

        elevation = mosaic[0]

    finally:
        for src in datasets:
            src.close()

    elevation = elevation.astype(np.float32)

    # Convert SRTM NoData to NaN.
    elevation[elevation <= -32768] = np.nan

    print("\nMosaic created successfully")
    print("Shape:", elevation.shape)
    print("CRS: EPSG:4326")
    print("Elevation min:", np.nanmin(elevation))
    print("Elevation max:", np.nanmax(elevation))
    print("Valid pixels:", np.isfinite(elevation).sum())

    return elevation, transform
    hgt_files = sorted(SRTM_DIR.glob("*.hgt"))

    if len(hgt_files) != 42:
        raise RuntimeError(
            f"Expected 42 HGT tiles, found {len(hgt_files)}"
        )

    arrays = []
    transforms = []

    for path in hgt_files:
        print(f"Reading {path.name}")

        data, transform = read_hgt(path)

        arrays.append(data)
        transforms.append(transform)

    # Determine mosaic extent from tile names.
    min_lon = min(
        (
            int(p.stem[4:]) if "E" in p.stem
            else -int(p.stem[4:])
        )
        for p in hgt_files
    )

    max_lon = max(
        (
            int(p.stem[4:]) if "E" in p.stem
            else -int(p.stem[4:])
        )
        for p in hgt_files
    )

    min_lat = min(
        (
            int(p.stem[1:3]) if p.stem[0] == "N"
            else -int(p.stem[1:3])
        )
        for p in hgt_files
    )

    max_lat = max(
        (
            int(p.stem[1:3]) if p.stem[0] == "N"
            else -int(p.stem[1:3])
        )
        for p in hgt_files
    )

    width = (max_lon - min_lon + 1) * 3600 + 1
    height = (max_lat - min_lat + 1) * 3600 + 1

    # We know the project area is only ~75-81E / 28-33N.
    # Build the mosaic directly from individual arrays rather
    # than relying on GDAL's HGT driver.
    mosaic = np.full(
        (height, width),
        np.nan,
        dtype=np.float32,
    )

    for path, data in zip(hgt_files, arrays):

        name = path.stem

        lat = int(name[1:3]) if name[0] == "N" else -int(name[1:3])

        if "E" in name:
            lon = int(name[name.find("E") + 1 :])
        else:
            lon = -int(name[name.find("W") + 1 :])

        row = (max_lat - lat - 1) * 3600
        col = (lon - min_lon) * 3600

        mosaic[row : row + 3601, col : col + 3601] = data

    transform = from_origin(
        min_lon,
        max_lat + 1,
        1 / 3600,
        1 / 3600,
    )

    print("\nMosaic created")
    print("Shape:", mosaic.shape)
    print("Elevation min:", np.nanmin(mosaic))
    print("Elevation max:", np.nanmax(mosaic))

    return mosaic, transform


# ============================================================
# SLOPE
# ============================================================

def calculate_slope(elevation, transform):
    """
    Calculate slope in degrees using local elevation gradients.

    The calculation accounts for geographic latitude because
    longitude/latitude pixels do not have equal physical width.
    """

    pixel_size_deg = abs(transform.a)

    height, width = elevation.shape

    # Approximate latitude at each row.
    latitudes = (
        transform.f
        - (np.arange(height) + 0.5) * pixel_size_deg
    )

    meters_per_degree_lat = 111_320.0

    meters_per_degree_lon = (
        111_320.0
        * np.cos(np.deg2rad(latitudes))
    )

    dy = pixel_size_deg * meters_per_degree_lat

    # np.gradient works row/column wise.
    grad_y, grad_x_deg = np.gradient(
        elevation,
        axis=(0, 1),
    )

    # Convert longitude gradient from elevation/degree
    # to elevation/metre.
    grad_x = np.empty_like(grad_x_deg)

    for row in range(height):
        dx = pixel_size_deg * meters_per_degree_lon[row]
        grad_x[row] = grad_x_deg[row] / dx

    grad_y = grad_y / dy

    slope_rad = np.arctan(
        np.sqrt(
            grad_x ** 2
            + grad_y ** 2
        )
    )

    return np.degrees(slope_rad)


# ============================================================
# CATCHMENT EXTRACTION
# ============================================================

def extract_features(
    elevation,
    transform,
    catchments,
):
    records = []

    print("\nExtracting terrain features...")

    for index, row in catchments.iterrows():

        catchment_id = str(
            row.get("HYBAS_ID", index)
        )

        geom = row.geometry

        # Bounding window first: avoids processing the whole
        # Himalayan mosaic for every catchment.
        minx, miny, maxx, maxy = geom.bounds

        col_start = max(
            0,
            int(
                (minx - transform.c)
                / transform.a
            ),
        )

        col_end = min(
            elevation.shape[1],
            int(
                (maxx - transform.c)
                / transform.a
            ) + 1,
        )

        row_start = max(
            0,
            int(
                (transform.f - maxy)
                / abs(transform.e)
            ),
        )

        row_end = min(
            elevation.shape[0],
            int(
                (transform.f - miny)
                / abs(transform.e)
            ) + 1,
        )

        if col_end <= col_start or row_end <= row_start:
            continue

        subset = elevation[
            row_start:row_end,
            col_start:col_end,
        ]

        subset_transform = rasterio.Affine(
            transform.a,
            0,
            transform.c + col_start * transform.a,
            0,
            transform.e,
            transform.f + row_start * transform.e,
        )

        # Create mask for pixels whose centres fall inside catchment.
        mask = geometry_mask(
            [geom],
            out_shape=subset.shape,
            transform=subset_transform,
            invert=True,
            all_touched=False,
        )

        values = subset[mask]

        values = values[np.isfinite(values)]

        if values.size == 0:
            print(
                f"WARNING: no valid elevation pixels for {catchment_id}"
            )
            continue

        # ----------------------------------------------------
        # Elevation statistics
        # ----------------------------------------------------

        mean_elevation = float(np.mean(values))
        min_elevation = float(np.min(values))
        max_elevation = float(np.max(values))
        elevation_std = float(np.std(values))
        relief = max_elevation - min_elevation

        # ----------------------------------------------------
        # Slope
        # ----------------------------------------------------

        # For slope, calculate only on the catchment window.
        slope = calculate_slope(
            subset,
            subset_transform,
        )

        slope_values = slope[mask]
        slope_values = slope_values[np.isfinite(slope_values)]

        if slope_values.size == 0:
            mean_slope = np.nan
            slope_std = np.nan
            max_slope = np.nan
        else:
            mean_slope = float(np.mean(slope_values))
            slope_std = float(np.std(slope_values))
            max_slope = float(np.max(slope_values))

        records.append(
            {
                "catchment_id": catchment_id,
                "mean_elevation_m": mean_elevation,
                "min_elevation_m": min_elevation,
                "max_elevation_m": max_elevation,
                "elevation_std_m": elevation_std,
                "relief_m": relief,
                "mean_slope_deg": mean_slope,
                "slope_std_deg": slope_std,
                "max_slope_deg": max_slope,
                "valid_elevation_pixels": int(values.size),
            }
        )

        print(
            f"{len(records):3d}/{len(catchments)} "
            f"{catchment_id} | "
            f"elev={mean_elevation:.1f} m | "
            f"relief={relief:.1f} m | "
            f"slope={mean_slope:.2f}°"
        )

    return pd.DataFrame(records)


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 60)
    print("FLASH-FLOOD PATTERN ATLAS")
    print("SRTM TERRAIN FEATURE EXTRACTION")
    print("=" * 60)

    # Load final 115 catchments.
    catchments = gpd.read_file(CATCHMENT_FILE)

    print("\nCatchments:", len(catchments))
    print("CRS:", catchments.crs)

    # Ensure geographic CRS for matching SRTM.
    if catchments.crs.to_epsg() != 4326:
        catchments = catchments.to_crs(4326)

    # Build SRTM mosaic.
    elevation, transform = build_mosaic()

    # Extract features.
    features = extract_features(
        elevation,
        transform,
        catchments,
    )

    # Save.
    features.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 60)
    print("TERRAIN EXTRACTION COMPLETE")
    print("=" * 60)

    print("Output:", OUTPUT_FILE)
    print("Rows:", len(features))
    print("Columns:", len(features.columns))

    print("\nMissing values:")
    print(features.isna().sum())

    print("\nPreview:")
    print(features.head())

    print("\nFeature summary:")
    print(
        features[
            [
                "mean_elevation_m",
                "relief_m",
                "mean_slope_deg",
            ]
        ].describe()
    )


if __name__ == "__main__":
    main()