import geopandas as gpd
import math
from pathlib import Path

file = Path(
    "data/processed/final_catchments_L8_50km2.gpkg"
)

gdf = gpd.read_file(file)

# Make sure we're working in geographic coordinates
gdf = gdf.to_crs("EPSG:4326")

minx, miny, maxx, maxy = gdf.total_bounds

print("Catchment extent:")
print(f"Longitude: {minx:.4f} to {maxx:.4f}")
print(f"Latitude : {miny:.4f} to {maxy:.4f}")

# SRTM 1-degree tile indices
west = math.floor(minx)
east = math.floor(maxx)
south = math.floor(miny)
north = math.floor(maxy)

tiles = []

for lat in range(south, north + 1):
    for lon in range(west, east + 1):

        lat_code = f"N{lat:02d}" if lat >= 0 else f"S{abs(lat):02d}"
        lon_code = f"E{lon:03d}" if lon >= 0 else f"W{abs(lon):03d}"

        tiles.append(f"{lat_code}{lon_code}")

print()
print("Required SRTM tiles:")
print("-" * 30)

for tile in tiles:
    print(tile)

print()
print("Total tiles:", len(tiles))