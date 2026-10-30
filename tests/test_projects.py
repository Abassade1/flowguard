"""Tests for the project model and SQLite store."""

import pytest

from modules import projects as store
from modules.projects import Project, validate_project


def test_sample_project_is_valid():
    p = store.load_sample_project()
    assert validate_project(p) == []
    assert 100 < p.straight_line_km < 150


def test_validation_catches_problems():
    assert validate_project(Project(name=" "))  # empty name
    assert validate_project(Project(start_lat=53.0, start_lon=-112.0, end_lat=53.0, end_lon=-112.0))  # too close
    assert validate_project(Project(start_lat=95))  # out of range
    assert validate_project(Project(budget=-1))


def test_save_load_update_delete_roundtrip(tmp_path):
    db = tmp_path / "t.db"
    p = Project(name="Line A", assumptions=[{"assumption": "x", "status": "Assumed", "source": "me"}])
    saved = store.save_project(p, db)
    assert saved.id is not None and saved.created_at
    saved.name = "Line A (rev 2)"
    store.save_project(saved, db)
    rows = store.list_projects(db)
    assert len(rows) == 1 and rows[0]["name"] == "Line A (rev 2)"
    loaded = store.load_project(saved.id, db)
    assert loaded.assumptions[0]["source"] == "me"
    store.delete_project(saved.id, db)
    assert store.list_projects(db) == []
    with pytest.raises(KeyError):
        store.load_project(saved.id, db)


def test_layers_are_deleted_with_their_project(tmp_path):
    db = tmp_path / "t.db"
    saved = store.save_project(Project(name="P"), db)
    store.save_layer(saved.id, "wetland", "W", "Test source", '{"type":"FeatureCollection","features":[]}', db_path=db)
    assert len(store.list_layers(saved.id, db)) == 1
    store.delete_project(saved.id, db)
    assert store.list_layers(saved.id, db) == []


def test_unknown_fields_are_ignored():
    p = Project.from_dict({"name": "Old file", "obsolete_field": 1})
    assert p.name == "Old file"


def test_units_and_currency():
    assert store.to_length(1.609344, "mi") == pytest.approx(1.0)
    p = Project(currency="USD", fx_rate_from_cad=0.7)
    assert store.to_currency(100.0, p) == pytest.approx(70.0)
    assert store.to_currency(100.0, Project()) == 100.0
    assert store.fmt_money(2.5e9, "CAD") == "CAD 2.50 B"
