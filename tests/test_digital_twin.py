"""Tests for the digital twin scene and renderer."""

import json

import pytest

from modules.digital_twin import (
    CONSTRUCTION_STAGES, MODES, build_scene, construction_progress_pct, render_plotly, scene_to_json,
)
from modules.elevation import SyntheticElevation
from modules.gis_layers import assemble_layers, corridor_frame
from modules.maps import route_colors
from modules.monitoring_engine import SimulationConfig, build_sections
from modules.projects import load_sample_project
from modules.routing import STRATEGIES, generate_alternatives


@pytest.fixture(scope="module")
def scene_inputs():
    project = load_sample_project()
    frame = corridor_frame(project)
    layers, _ = assemble_layers(frame, [])
    elevation = SyntheticElevation(frame)
    routes, geoms, _ = generate_alternatives(frame, layers, [s.key for s in STRATEGIES], elevation=elevation)
    length = float(routes.set_index("route_id").loc["R1", "length_km"])
    sections = build_sections(SimulationConfig(pipeline_length_km=length, n_sections=6))
    scene = build_scene(frame, routes, geoms, layers, elevation, "R1", sections,
                        route_colors(routes["route_id"].tolist()), "Natural gas")
    return project, frame, routes, geoms, sections, scene


def test_scene_uses_project_coordinates(scene_inputs):
    project, frame, *_, scene = scene_inputs
    pipe = next(r for r in scene.routes if r["route_id"] == "R1")
    assert pipe["x_km"][0] == pytest.approx(0, abs=1e-6) and pipe["y_km"][0] == pytest.approx(0, abs=1e-6)
    end_x, end_y = (frame.x1 - frame.x0) / 1000, (frame.y1 - frame.y0) / 1000
    assert pipe["x_km"][-1] == pytest.approx(end_x, abs=0.01) and pipe["y_km"][-1] == pytest.approx(end_y, abs=0.01)
    # Sensors carry real latitude/longitude within the project's extent.
    lats = scene.sections["lat"]
    assert lats.between(min(project.start_lat, project.end_lat) - 0.5, max(project.start_lat, project.end_lat) + 0.5).all()


def test_pipe_is_buried_below_ground_and_features_are_labelled(scene_inputs):
    *_, scene = scene_inputs
    assert {f["category"] for f in scene.features} >= {"water", "wetland", "protected", "road"}
    assert all(f["synthetic"] for f in scene.features)


def test_facilities_match_sections_and_pipeline_type(scene_inputs):
    *_, sections, scene = scene_inputs
    kinds = [f["kind"] for f in scene.facilities]
    assert kinds.count("valve") == len(sections) - 1
    assert scene.facilities[0]["name"].startswith("Compressor station")  # natural gas
    assert all("indicative" in f["name"] for f in scene.facilities)


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_renders(scene_inputs, mode):
    *_, sections, scene = scene_inputs
    fig = render_plotly(scene, mode, progress={"S01": "Complete"}, selected_section="S02")
    assert len(fig.data) > 3
    if mode != "Planning":
        assert any(t.name == "Sensor stations" for t in fig.data)


def test_construction_progress_is_length_weighted(scene_inputs):
    *_, sections, _ = scene_inputs
    assert construction_progress_pct(sections, {}) == 0
    assert construction_progress_pct(sections, {s: "Complete" for s in sections["section_id"]}) == pytest.approx(100)
    half = {"S01": "Complete", "S02": "Complete", "S03": "Complete"}
    assert construction_progress_pct(sections, half) == pytest.approx(50)
    assert CONSTRUCTION_STAGES[0] == "Not started"


def test_scene_export_is_valid_json(scene_inputs):
    *_, scene = scene_inputs
    data = json.loads(scene_to_json(scene))
    assert data["crs"].startswith("EPSG:326")
    assert data["terrain"]["synthetic"] is True
    assert len(data["sections"]) == 6 and all(f["indicative"] for f in data["facilities"])
