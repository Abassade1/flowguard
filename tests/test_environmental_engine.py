"""Tests for environmental screening."""

import json

import pytest

from modules.environmental_engine import (
    LEVELS, THEMES, conflicts, confidence_for, level_matrix, route_summary, screen_routes,
)
from modules.gis_layers import assemble_layers, corridor_frame
from modules.projects import load_sample_project
from modules.routing import STRATEGIES, generate_alternatives


@pytest.fixture(scope="module")
def demo():
    project = load_sample_project()
    frame = corridor_frame(project)
    layers, _ = assemble_layers(frame, [])
    routes, geoms, _ = generate_alternatives(frame, layers, [s.key for s in STRATEGIES])
    return project, frame, layers, routes, geoms, screen_routes(frame, routes, geoms, layers)


def test_every_route_gets_every_theme(demo):
    *_, routes, _, screening = demo
    assert len(screening) == len(routes) * len(THEMES)
    assert set(screening["level"]) <= set(LEVELS)


def test_missing_habitat_is_not_assessed_never_low(demo):
    *_, screening = demo
    habitat = screening[screening["theme"] == "habitat"]
    assert (habitat["level"] == "Not assessed").all()
    assert (habitat["confidence"] == "Not assessed").all()
    assert habitat["explanation"].str.contains("unknown, not absent").all()


def test_synthetic_results_have_no_confidence(demo):
    *_, screening = demo
    wet = screening[screening["theme"] == "wetlands"]
    assert wet["confidence"].str.startswith("None").all()


def test_reference_line_is_high_for_protected_area(demo):
    *_, screening = demo
    row = screening[(screening["route_id"] == "REF") & (screening["theme"] == "protected")].iloc[0]
    assert row["level"] == "High" and row["value"] > 0


def test_environment_strategy_has_fewer_high_flags_than_direct(demo):
    *_, routes, _, screening = demo
    rid = dict(zip(routes["strategy"], routes["route_id"]))
    high = screening[screening["level"] == "High"].groupby("route_id").size()
    assert high.get(rid["environment"], 0) < high.get(rid["direct"], 0)


def test_matrix_and_summary_shapes(demo):
    *_, routes, _, screening = demo
    m = level_matrix(screening)
    assert m.shape == (len(THEMES), len(routes))
    s = route_summary(screening)
    assert (s[LEVELS].sum(axis=1) == len(THEMES)).all()


def test_conflicts_name_specific_features(demo):
    _, _, layers, _, geoms, _ = demo
    found = conflicts(geoms["REF"], layers)
    assert found["feature"].str.contains("protected area PA-1").any()
    assert (found["source"].str.startswith("SYNTHETIC")).all()


def test_user_habitat_enables_theme(demo):
    project, frame, _, _, _, _ = demo
    lat = (project.start_lat + project.end_lat) / 2
    lon = (project.start_lon + project.end_lon) / 2
    square = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"name": "Calving area"},
              "geometry": {"type": "Polygon", "coordinates": [[[lon - .3, lat - .02], [lon + .3, lat - .02],
                                                                [lon + .3, lat + .02], [lon - .3, lat + .02],
                                                                [lon - .3, lat - .02]]]}}]}
    row = {"category": "habitat", "name": "Habitat", "source": "Test", "geojson": json.dumps(square)}
    layers, _ = assemble_layers(frame, [row])
    routes, geoms, _ = generate_alternatives(frame, layers, ["direct"], avoid_hard_constraints=True)
    screening = screen_routes(frame, routes, geoms, layers)
    habitat = screening[screening["theme"] == "habitat"]
    assert (habitat["level"] != "Not assessed").all()
    assert confidence_for(next(t for t in THEMES if t.key == "habitat"), layers)[0] == "User-supplied"
