"""Tests for GIS layers and corridor generation."""

import json

import numpy as np
import pytest
from shapely.geometry import LineString, mapping

from modules.gis_layers import (
    assemble_layers, corridor_frame, prepare_user_layer, read_uploaded_layer, synthetic_layers, utm_crs_for,
)
from modules.projects import load_sample_project
from modules.routing import (
    STRATEGIES, build_cost_grid, generate_alternatives, least_cost_path, measure_route,
)


@pytest.fixture(scope="module")
def setup():
    project = load_sample_project()
    frame = corridor_frame(project)
    layers, register = assemble_layers(frame, [])
    return project, frame, layers, register


def test_utm_zone_selection():
    assert utm_crs_for(-112.5, 53.0).to_epsg() == 32612
    assert utm_crs_for(151.2, -33.9).to_epsg() == 32756


def test_frame_round_trip_and_length(setup):
    project, frame, _, _ = setup
    u, v = frame.to_uv(*frame.to_xy(5000.0, -2000.0))
    assert (float(u), float(v)) == pytest.approx((5000.0, -2000.0))
    # Projected length agrees with the geodesic distance to within 0.5 %.
    assert frame.length_m / 1000 == pytest.approx(project.straight_line_km, rel=0.005)


def test_synthetic_layers_are_labelled(setup):
    _, frame, _, register = setup
    layers = synthetic_layers(frame)
    for cat, gdf in layers.items():
        if cat == "habitat":
            assert gdf.empty  # never invented: habitat stays a data gap without real data
            continue
        assert gdf["synthetic"].all()
        assert gdf["name"].str.contains("SYNTHETIC").all()
    by_category = {r["category"]: r["data_type"] for r in register}
    assert by_category.pop("Sensitive wildlife habitat") == "Not available"
    assert set(by_category.values()) == {"Synthetic demonstration"}


def test_least_cost_path_avoids_barrier():
    cost = np.ones((5, 5))
    cost[1:5, 2] = np.inf  # wall with a gap at the top row
    path = least_cost_path(cost, (4, 0), (4, 4))
    assert path[0] == (4, 0) and path[-1] == (4, 4)
    assert (0, 2) in path
    cost[0, 2] = np.inf
    assert least_cost_path(cost, (4, 0), (4, 4)) is None


def test_generated_routes_respect_hard_constraints(setup):
    _, frame, layers, _ = setup
    routes, geoms, _ = generate_alternatives(frame, layers, [s.key for s in STRATEGIES], avoid_hard_constraints=True)
    ref = routes.set_index("route_id").loc["REF"]
    assert ref["protected_km"] > 0 and ref["hard_constraints"]
    generated = routes[routes["route_id"] != "REF"]
    assert len(generated) >= 3
    assert (generated["protected_km"] == 0).all()
    assert (generated["length_km"] >= ref["length_km"]).all()  # nothing is shorter than a straight line


def test_environment_strategy_reduces_wetland(setup):
    _, frame, layers, _ = setup
    routes, _, _ = generate_alternatives(frame, layers, ["direct", "environment"])
    r = routes.set_index("strategy")
    assert r.loc["environment", "wetland_km"] < r.loc["direct", "wetland_km"]


def test_measure_route_counts_crossings(setup):
    _, frame, layers, _ = setup
    straight = LineString([(frame.x0, frame.y0), (frame.x1, frame.y1)])
    m = measure_route(frame, straight, layers)
    assert m["water_crossings"] >= 3
    assert m["length_km"] == pytest.approx(frame.length_m / 1000)


def test_user_layer_replaces_synthetic(setup):
    project, frame, _, _ = setup
    # A user protected area: a small square around the route midpoint, in lat/lon.
    lat = (project.start_lat + project.end_lat) / 2
    lon = (project.start_lon + project.end_lon) / 2
    square = {"type": "Polygon", "coordinates": [[[lon - .05, lat - .05], [lon + .05, lat - .05],
                                                  [lon + .05, lat + .05], [lon - .05, lat + .05], [lon - .05, lat - .05]]]}
    fc = {"type": "FeatureCollection",
          "features": [{"type": "Feature", "properties": {"name": "Test park"}, "geometry": square}]}
    gdf, warnings = read_uploaded_layer("park.geojson", json.dumps(fc).encode())
    assert warnings == [] and len(gdf) == 1
    prepared, _ = prepare_user_layer(gdf, "protected", frame, "Parks", "Unit test")
    assert prepared["name"].iloc[0] == "Test park" and not prepared["synthetic"].any()

    row = {"category": "protected", "name": "Parks", "source": "Unit test", "geojson": json.dumps(fc)}
    layers, register = assemble_layers(frame, [row])
    assert list(layers["protected"]["name"]) == ["Test park"]
    assert any(r["data_type"] == "User-supplied" for r in register)


def test_upload_rejects_unknown_format():
    with pytest.raises(ValueError):
        read_uploaded_layer("data.csv", b"a,b\n1,2")


def test_line_layer_rejected_for_polygon_category(setup):
    _, frame, _, _ = setup
    project = load_sample_project()
    line = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": mapping(
        LineString([(project.start_lon, project.start_lat), (project.end_lon, project.end_lat)]))}]}
    gdf, _ = read_uploaded_layer("l.geojson", json.dumps(line).encode())
    prepared, warnings = prepare_user_layer(gdf, "wetland", frame, "W", "test")
    assert prepared.empty and warnings


def test_cost_grid_marks_constraints(setup):
    _, frame, layers, _ = setup
    grid = build_cost_grid(frame, layers)
    assert grid.masks["protected"].any() and grid.masks["water"].any()
    assert grid.inside.sum() > 1000
