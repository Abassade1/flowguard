"""Elevation sources: synthetic terrain or a user-supplied digital elevation model (DEM).

Every part of the app that needs elevation (routing slopes, profiles, environmental
proxies, the 3D twin) asks an `ElevationSource` for heights at projected x/y
coordinates. Two implementations exist:

* `SyntheticElevation` - the formula-based demo terrain (not real).
* `DemElevation`       - a GeoTIFF DEM uploaded by the user (e.g. NRCan MRDEM/CDEM,
                         Copernicus GLO-30). It is reprojected onto the project's
                         UTM grid and sampled with bilinear interpolation.

Where the DEM has no data (or does not reach), heights are NaN: unknown, never guessed.

Storage: on upload, the DEM is cropped to the project's study area (plus a margin)
and saved as a compressed GeoTIFF under data/dem/project_<id>/. Only the crop is kept.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import shapely
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds

from modules.gis_layers import CorridorFrame, synthetic_elevation
from modules.projects import DEFAULT_DB_PATH, ROOT, connect, init_db

DEM_DIR = ROOT / "data" / "dem"
MAX_STORED_CELLS = 2500  # longest side of the stored crop (downsampled beyond this)
MIN_CELL_M = 30.0  # finest sampling grid used inside the app
GRID_CELLS = 500  # cells along the longest side of the in-app sampling grid


class SyntheticElevation:
    is_synthetic = True
    label = "Synthetic elevation surface (not real)"
    coverage = 1.0

    def __init__(self, frame: CorridorFrame):
        self.frame = frame

    def sample(self, x, y) -> np.ndarray:
        return synthetic_elevation(self.frame, x, y)


@dataclass
class DemElevation:
    """A DEM resampled onto a regular grid in the project's projected CRS (metres)."""

    grid: np.ndarray  # rows from north to south; NaN = no data
    left: float
    top: float
    cell: float
    label: str
    coverage: float  # share of the study area with valid heights (0-1)
    is_synthetic = False

    @classmethod
    def from_file(cls, path: Path, frame: CorridorFrame, label: str) -> "DemElevation":
        area = frame.study_area()
        minx, miny, maxx, maxy = area.bounds
        cell = max(MIN_CELL_M, max(maxx - minx, maxy - miny) / GRID_CELLS)
        width, height = int(np.ceil((maxx - minx) / cell)), int(np.ceil((maxy - miny) / cell))
        dst = np.full((height, width), np.nan, dtype="float32")
        with rasterio.open(path) as src:
            reproject(
                source=rasterio.band(src, 1), destination=dst,
                dst_transform=from_origin(minx, maxy, cell, cell), dst_crs=f"EPSG:{frame.crs_epsg}",
                resampling=Resampling.bilinear, dst_nodata=np.nan,
            )
        # Coverage: fraction of study-area grid cells that have a height.
        cx = minx + (np.arange(width) + 0.5) * cell
        cy = maxy - (np.arange(height) + 0.5) * cell
        gx, gy = np.meshgrid(cx, cy)
        inside = shapely.contains_xy(area, gx, gy)
        coverage = float(np.isfinite(dst[inside]).mean()) if inside.any() else 0.0
        return cls(dst, minx, maxy, cell, label, coverage)

    def sample(self, x, y) -> np.ndarray:
        """Bilinear interpolation; NaN outside the grid or next to missing cells."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        col = (x - self.left) / self.cell - 0.5
        row = (self.top - y) / self.cell - 0.5
        c0, r0 = np.floor(col).astype(int), np.floor(row).astype(int)
        fc, fr = col - c0, row - r0
        h, w = self.grid.shape
        valid = (c0 >= 0) & (r0 >= 0) & (c0 + 1 < w) & (r0 + 1 < h)
        c0c, r0c = np.clip(c0, 0, w - 2), np.clip(r0, 0, h - 2)
        g = self.grid
        z = (g[r0c, c0c] * (1 - fc) * (1 - fr) + g[r0c, c0c + 1] * fc * (1 - fr)
             + g[r0c + 1, c0c] * (1 - fc) * fr + g[r0c + 1, c0c + 1] * fc * fr)
        return np.where(valid, z, np.nan)


# ---------------------------------------------------------------------------
# Upload, storage and lookup
# ---------------------------------------------------------------------------
def save_dem(project_id: int, frame: CorridorFrame, filename: str, content: bytes, source: str,
             licence: str = "", notes: str = "", db_path: Path = DEFAULT_DB_PATH,
             dem_dir: Path = DEM_DIR) -> dict:
    """Crop an uploaded GeoTIFF to the study area, store it, and record it. Returns the record."""
    if Path(filename).suffix.lower() not in (".tif", ".tiff"):
        raise ValueError("Please upload a GeoTIFF (.tif / .tiff) elevation file.")
    area = frame.study_area().buffer(frame.length_m * 0.10)
    with MemoryFile(content) as mem, mem.open() as src:
        if src.crs is None:
            raise ValueError("The GeoTIFF has no coordinate system, so it cannot be placed on the map.")
        bounds = transform_bounds(f"EPSG:{frame.crs_epsg}", src.crs, *area.bounds, densify_pts=21)
        window = from_bounds(*bounds, transform=src.transform)
        full = Window(0, 0, src.width, src.height)
        try:
            window = window.intersection(full).round_offsets().round_lengths()
        except Exception as exc:  # rasterio raises when windows do not overlap
            raise ValueError("The elevation file does not cover this project's area.") from exc
        if window.width < 2 or window.height < 2:
            raise ValueError("The elevation file does not cover this project's area.")
        scale = max(1.0, max(window.width, window.height) / MAX_STORED_CELLS)
        out_w, out_h = int(window.width / scale), int(window.height / scale)
        data = src.read(1, window=window, out_shape=(out_h, out_w), masked=True,
                        resampling=Resampling.bilinear).astype("float32").filled(np.nan)
        transform = src.window_transform(window) @ rasterio.Affine.scale(window.width / out_w,
                                                                         window.height / out_h)
        crs = src.crs
    if not np.isfinite(data).any():
        raise ValueError("The elevation file has no valid heights inside this project's area.")

    folder = Path(dem_dir) / f"project_{project_id}"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"dem_{datetime.now():%Y%m%d_%H%M%S}.tif"
    with rasterio.open(path, "w", driver="GTiff", width=out_w, height=out_h, count=1, dtype="float32",
                       crs=crs, transform=transform, nodata=np.nan, compress="deflate") as dst:
        dst.write(data, 1)

    init_db(db_path)
    now = datetime.now().isoformat(timespec="seconds")
    with connect(db_path) as conn:
        conn.execute("DELETE FROM dems WHERE project_id = ?", (project_id,))  # one DEM per project
        cur = conn.execute(
            "INSERT INTO dems (project_id, name, source, licence, notes, file_path, loaded_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)", (project_id, filename, source, licence, notes, str(path), now))
    return {"id": cur.lastrowid, "project_id": project_id, "name": filename, "source": source, "licence": licence,
            "notes": notes, "file_path": str(path), "loaded_at": now}


def get_dem(project_id: int | None, db_path: Path = DEFAULT_DB_PATH) -> dict | None:
    if project_id is None:
        return None
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute("SELECT * FROM dems WHERE project_id = ? ORDER BY id DESC LIMIT 1",
                           (project_id,)).fetchone()
    return dict(row) if row and Path(row["file_path"]).exists() else None


def delete_dem(project_id: int, db_path: Path = DEFAULT_DB_PATH) -> None:
    record = get_dem(project_id, db_path)
    with connect(db_path) as conn:
        conn.execute("DELETE FROM dems WHERE project_id = ?", (project_id,))
    if record:
        Path(record["file_path"]).unlink(missing_ok=True)


def elevation_for(frame: CorridorFrame, dem_record: dict | None):
    """The elevation source to use: the project's DEM if it has one, else synthetic terrain."""
    if dem_record is None:
        return SyntheticElevation(frame)
    return DemElevation.from_file(Path(dem_record["file_path"]), frame,
                                  f"{dem_record['name']} ({dem_record['source']})")


def register_entry(elevation) -> dict:
    """Row for the data-source register."""
    if elevation.is_synthetic:
        return {"category": "Terrain / elevation", "layer": "Synthetic elevation surface",
                "source": "SYNTHETIC: generated by FlowGuard (not real)", "licence": "n/a",
                "features": 0, "data_type": "Synthetic demonstration", "confidence": "None: not real data",
                "notes": "Upload a GeoTIFF DEM on Route Planning to use real terrain."}
    return {"category": "Terrain / elevation", "layer": elevation.label, "source": elevation.label,
            "licence": "", "features": 0, "data_type": "User-supplied DEM",
            "confidence": f"Medium: {elevation.coverage:.0%} of study area covered; resampled to "
                          f"{elevation.cell:,.0f} m cells",
            "notes": "Areas without DEM coverage have unknown elevation (not estimated)."}
