import earthaccess
from pathlib import Path

# --------------------------------------------------
# Study-area extent from our 115 final catchments
# --------------------------------------------------

bbox = (
    75.5663,   # west
    28.4786,   # south
    81.1087,   # east
    33.2547    # north
)

# --------------------------------------------------
# Output directory
# --------------------------------------------------

output_dir = Path("data/raw/srtm")
output_dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Login
# --------------------------------------------------

earthaccess.login()

# --------------------------------------------------
# Search SRTMGL1 V003
# --------------------------------------------------

print("Searching NASA SRTMGL1 V003...")

results = earthaccess.search_data(
    short_name="SRTMGL1",
    version="003",
    bounding_box=bbox
)

print()
print("Granules found:", len(results))

# --------------------------------------------------
# Download
# --------------------------------------------------

if len(results) == 0:
    raise RuntimeError("No SRTM granules found.")

print()
print("Downloading SRTM tiles...")

files = earthaccess.download(
    results,
    local_path=str(output_dir)
)

print()
print("Download complete.")
print("Files downloaded:", len(files))
print()
print("Output directory:")
print(output_dir)