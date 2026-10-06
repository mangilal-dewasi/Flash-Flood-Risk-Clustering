import geopandas as gpd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
hydro_file = Path(
    "data/hydrobasins/hybas_as_lev08_v1c.shp"
)

boundary_file = Path(
    "data/interim/western_himalaya_study_boundary.shp"
)

output_dir = Path("data/interim")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = (
    output_dir / "western_himalaya_catchments_L8.gpkg"
)

# -----------------------------
# Load data
# -----------------------------
print("Loading HydroBASINS L8...")
basins = gpd.read_file(hydro_file)

print("Loading study boundary...")
study = gpd.read_file(boundary_file)

# Merge HP + Uttarakhand
study_geom = study.geometry.union_all()

# -----------------------------
# Equal-area CRS
# -----------------------------
area_crs = "EPSG:6933"

basins_area = basins.to_crs(area_crs)

study_area = gpd.GeoDataFrame(
    {"geometry": [study_geom]},
    crs="EPSG:4326"
).to_crs(area_crs)

study_geom_area = study_area.geometry.iloc[0]

# -----------------------------
# Calculate overlap
# -----------------------------
print("Calculating catchment overlap...")

basins_area["catchment_area_km2"] = (
    basins_area.geometry.area / 1_000_000
)

basins_area["intersection_area_km2"] = (
    basins_area.geometry
    .intersection(study_geom_area)
    .area / 1_000_000
)

basins_area["study_overlap_pct"] = (
    basins_area["intersection_area_km2"]
    / basins_area["catchment_area_km2"]
    * 100
)

# -----------------------------
# Select final catchments
# -----------------------------
final = basins_area[
    basins_area["study_overlap_pct"] >= 50
].copy()

print()
print("Final catchments:", len(final))

# -----------------------------
# Return to geographic CRS
# -----------------------------
final = final.to_crs("EPSG:4326")

# -----------------------------
# Save GeoPackage
# -----------------------------
final.to_file(
    output_file,
    layer="catchments",
    driver="GPKG"
)

print()
print("Saved:")
print(output_file)

print()
print("Catchment area statistics:")
print(
    final["catchment_area_km2"]
    .describe()
)

print()
print("Overlap statistics:")
print(
    final["study_overlap_pct"]
    .describe()
)