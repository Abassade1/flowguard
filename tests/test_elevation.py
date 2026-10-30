"""Tests for DEM upload, storage and sampling (uses a small generated GeoTIFF with known heights)."""

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from modules.elevation import DemElevation, SyntheticElevation, delete_dem, elevation_for, get_dem, save_dem
from modules.gis_layers import assemble_layers, corridor_frame
from modules.projects import load_sample_project, save_project
from modules.routing import generate_alternatives


def make_geotiff(lon0, lat0, lon1, lat1, res=0.01, nodata_box=None) -> bytes:
    """Heights rise 100 m per degree of longitude: z = 500 + 100 * (lon - lon0)."""
    width, height = int((lon1 - lon0) / res), int((lat1 - lat0) / res)
    lons = lon0 + (np.arange(width) + 0.5) * res
    data = np.tile(500 + 100 * (lons - lon0), (height, 1)).astype("float32")
    if nodata_box:
        data[nodata_box] = -9999
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", width=width, height=height, count=1, dtype="float32", crs="EPSG:4326",
                      transform=from_origin(lon0, lat1, res, res), nodata=-9999) as dst:
            dst.write(data, 1)
        return mem.read()


@pytest.fixture
def project_and_frame(tmp_path):
    db = tmp_path / "t.db"
    # A ~130 km corridor inside the generated GeoTIFFs (independent of where the demo project is).
    project = load_sample_project()
    project.start_lat, project.start_lon, project.end_lat, project.end_lon = 53.20, -112.90, 52.55, -111.40
    project = save_project(project, db)
    return project, corridor_frame(project), db, tmp_path / "dem"


def test_upload_crop_and_sample_known_heights(project_and_frame):
    project, frame, db, dem_dir = project_and_frame
    tif = make_geotiff(-115.0, 51.0, -109.0, 55.0)
    record = save_dem(project.id, frame, "test.tif", tif, "Unit test", db_path=db, dem_dir=dem_dir)
    assert get_dem(project.id, db)["id"] == record["id"]
    with rasterio.open(record["file_path"]) as src:  # stored crop is smaller than the 600x400 input
        assert src.width < 600 and src.height < 400
    dem = elevation_for(frame, record)
    assert isinstance(dem, DemElevation) and not dem.is_synthetic
    assert dem.coverage == pytest.approx(1.0, abs=0.01)
    expected_start = 500 + 100 * (project.start_lon + 115.0)
    assert dem.sample([frame.x0], [frame.y0])[0] == pytest.approx(expected_start, abs=3.0)


def test_partial_coverage_gives_nan_not_guesses(project_and_frame):
    project, frame, db, dem_dir = project_and_frame
    # Only the western half of the corridor is covered.
    mid_lon = (project.start_lon + project.end_lon) / 2
    tif = make_geotiff(-115.0, 51.0, mid_lon, 55.0)
    dem = elevation_for(frame, save_dem(project.id, frame, "half.tif", tif, "Unit test", db_path=db, dem_dir=dem_dir))
    assert 0.2 < dem.coverage < 0.8
    assert np.isnan(dem.sample([frame.x1], [frame.y1])[0])
    layers, _ = assemble_layers(frame, [])
    routes, _, _ = generate_alternatives(frame, layers, ["direct"], elevation=dem)
    assert "elevation unknown" in routes.set_index("route_id").loc["R1", "geohazards"]


def test_rejects_files_that_do_not_fit(project_and_frame):
    project, frame, db, dem_dir = project_and_frame
    with pytest.raises(ValueError, match="does not cover"):
        save_dem(project.id, frame, "far.tif", make_geotiff(10.0, 40.0, 12.0, 42.0), "x", db_path=db, dem_dir=dem_dir)
    with pytest.raises(ValueError, match="GeoTIFF"):
        save_dem(project.id, frame, "dem.png", b"123", "x", db_path=db, dem_dir=dem_dir)


def test_delete_returns_to_synthetic(project_and_frame):
    project, frame, db, dem_dir = project_and_frame
    record = save_dem(project.id, frame, "t.tif", make_geotiff(-115.0, 51.0, -109.0, 55.0), "x", db_path=db,
                      dem_dir=dem_dir)
    delete_dem(project.id, db)
    assert get_dem(project.id, db) is None
    assert isinstance(elevation_for(frame, None), SyntheticElevation)
    from pathlib import Path
    assert not Path(record["file_path"]).exists()
