import geopandas as gpd
from pathlib import Path

file = Path(
    "data/interim/western_himalaya_catchments_L8.gpkg"
)

gdf = gpd.read_file(file)

area_crs = gdf.to_crs("EPSG:6933")

gdf["area_km2"] = (
    area_crs.geometry.area / 1_000_000
)

print("Total catchments:", len(gdf))
print()

print("Catchment area statistics:")
print(gdf["area_km2"].describe())
print()

thresholds = [10, 25, 50, 100, 250, 500]

print("Catchments retained at different minimum-area thresholds:")
print("-" * 55)

for threshold in thresholds:
    count = (gdf["area_km2"] >= threshold).sum()
    print(
        f">= {threshold:>3} km² : "
        f"{count:>3} catchments"
    )

print()
print("Smallest 15 catchments:")
print(
    gdf[["HYBAS_ID", "area_km2"]]
    .sort_values("area_km2")
    .head(15)
    .to_string(index=False)
)