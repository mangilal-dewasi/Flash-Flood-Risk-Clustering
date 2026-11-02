"""
Flash-Flood Pattern Atlas: download and prepare raw datasets.

Usage (from repository root):
    python download.py

The Google Drive files must be shared as "Anyone with the link — Viewer".
This script never deletes existing project data.
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

import gdown

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DOWNLOADS = ROOT / "downloads"

DATASETS = [
    {
        "name": "HydroBASINS",
        "id": "1FdOgLeZze_IjrEAtGlWAeV3BvA_u510N",
        "zip": DOWNLOADS / "hydrobasins_as.zip",
        "target": DATA / "hydrobasins",
        "kind": "hydro",
    },
    {
        "name": "Rainfall (2019–2024)",
        "id": "19j_QKbaXhKPgbaZF0CmCyZd2n3dVv-hj",
        "zip": DOWNLOADS / "rainfall_2019_2024.zip",
        "target": DATA,
        "kind": "rainfall",
    },
    {
        "name": "SRTM elevation tiles",
        "id": "1RQzv-eOUMtFa_puyVcxsR1O29NqT7Fqy",
        "zip": DOWNLOADS / "srtm_tiles.zip",
        "target": DATA / "raw" / "srtm",
        "kind": "srtm",
    },
]


def ready(item: dict) -> bool:
    target = item["target"]
    kind = item["kind"]

    if kind == "rainfall":
        return all((target / f"RF25_{year}.nc").is_file()
                   for year in range(2019, 2025))
    if kind == "hydro":
        # Need the shapefile and its DBF companion somewhere under the folder.
        return any(target.rglob("*.shp")) and any(target.rglob("*.dbf"))
    if kind == "srtm":
        return any(target.rglob("*.hgt"))
    return False


def download_zip(item: dict) -> Path:
    archive = item["zip"]
    archive.parent.mkdir(parents=True, exist_ok=True)

    if archive.is_file() and zipfile.is_zipfile(archive):
        print(f"Using existing archive: {archive.relative_to(ROOT)}")
        return archive

    # Discard only an incomplete/invalid downloaded archive, never source data.
    if archive.exists():
        archive.unlink()

    url = f"https://drive.google.com/uc?id={item['id']}"
    print(f"Downloading {item['name']}…")
    result = gdown.download(url=url, output=str(archive), quiet=False, fuzzy=True)

    if not result or not archive.is_file() or not zipfile.is_zipfile(archive):
        archive.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download failed or file is not a ZIP for {item['name']}. "
            "Check the Drive sharing setting and verify the shared file."
        )
    return archive


def safe_extract(archive_path: Path, staging: Path) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    root = staging.resolve()

    with zipfile.ZipFile(archive_path) as zf:
        bad_member = zf.testzip()
        if bad_member:
            raise RuntimeError(f"ZIP integrity check failed: {bad_member}")

        for member in zf.infolist():
            destination = (staging / member.filename).resolve()
            if destination != root and root not in destination.parents:
                raise RuntimeError(f"Unsafe ZIP member path: {member.filename}")
        zf.extractall(staging)


def install_extracted(item: dict, staging: Path) -> None:
    """Place archive contents at the expected paths, handling wrapper folders."""
    target = item["target"]
    kind = item["kind"]
    target.mkdir(parents=True, exist_ok=True)

    if kind == "rainfall":
        # Archives may contain annual NetCDFs directly or under data/ or a wrapper.
        for year in range(2019, 2025):
            candidates = list(staging.rglob(f"RF25_{year}.nc"))
            if not candidates:
                raise RuntimeError(f"Archive is missing RF25_{year}.nc")
            source = candidates[0]
            destination = target / source.name
            if not destination.exists():
                shutil.copy2(source, destination)

    elif kind == "srtm":
        tiles = list(staging.rglob("*.hgt"))
        if not tiles:
            raise RuntimeError("SRTM archive contains no .hgt tiles.")
        for source in tiles:
            destination = target / source.name
            if not destination.exists():
                shutil.copy2(source, destination)

    elif kind == "hydro":
        # Keep the archive's internal hierarchy, but strip a redundant outer
        # data/hydrobasins wrapper if present.
        shapefiles = list(staging.rglob("*.shp"))
        if not shapefiles:
            raise RuntimeError("HydroBASINS archive contains no .shp files.")

        # Copy the containing HydroBASINS directory tree into target.
        # Preserve sidecar files (.dbf/.shx/.prj/.cpg) alongside each shapefile.
        for shp in shapefiles:
            parent = shp.parent
            for source in parent.iterdir():
                if source.is_file():
                    dest = target / source.name
                    if not dest.exists():
                        shutil.copy2(source, dest)

        # If the ZIP contains nested levels, preserve any additional files/folders.
        # Above copies each shapefile's sidecars, which are the needed GIS inputs.


def prepare(item: dict) -> None:
    print(f"\n== {item['name']} ==")
    if ready(item):
        print(f"Already prepared at {item['target'].relative_to(ROOT)}; skipping.")
        return

    archive = download_zip(item)
    staging = DOWNLOADS / f"_extract_{item['kind']}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    try:
        safe_extract(archive, staging)
        install_extracted(item, staging)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    if not ready(item):
        raise RuntimeError(
            f"Files were extracted, but validation failed for {item['name']}. "
            "The archive layout may differ from the expected layout."
        )
    print(f"Ready: {item['target'].relative_to(ROOT)}")


def main() -> int:
    print("Flash-Flood Pattern Atlas — dataset setup")
    print(f"Repository root: {ROOT}")
    try:
        for item in DATASETS:
            prepare(item)
    except Exception as exc:
        print(f"\nSETUP FAILED: {exc}", file=sys.stderr)
        return 1

    print("\nDataset setup complete. Next run: python pipeline.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
