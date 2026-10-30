"""Tests for engagement records."""

from datetime import date

import pytest

from modules import social_impact as si
from modules.projects import Project, delete_project, save_project


@pytest.fixture
def db_project(tmp_path):
    db = tmp_path / "t.db"
    return db, save_project(Project(name="P"), db)


def test_add_list_update_delete(db_project):
    db, p = db_project
    rid = si.add_record(p.id, "concern", {"title": "Dust near homes", "community": "Residents"}, db)
    rows = si.list_records(p.id, db)
    assert rows.loc[0, "status"] == "Open"  # first status by default
    si.update_record(rid, {"status": "Responded", "response": "Watering plan"}, db)
    assert si.list_records(p.id, db).loc[0, "status"] == "Responded"
    si.delete_record(rid, db)
    assert si.list_records(p.id, db).empty


def test_title_required_and_sensitivity_checked(db_project):
    db, p = db_project
    with pytest.raises(ValueError):
        si.add_record(p.id, "activity", {"title": "  "}, db)
    with pytest.raises(ValueError):
        si.add_record(p.id, "activity", {"title": "x", "sensitivity": "Secret"}, db)


def test_landuse_defaults_to_restricted_and_has_no_location_fields(db_project):
    db, p = db_project
    si.add_record(p.id, "landuse", {"title": "Spring harvest", "details": "private"}, db)
    assert si.list_records(p.id, db).loc[0, "sensitivity"] == si.RESTRICTED
    fields = set(si.RECORD_TYPES["landuse"].fields)
    assert not fields & {"lat", "lon", "latitude", "longitude", "geometry", "location"}


def test_restricted_content_is_redacted_and_never_exported(db_project):
    db, p = db_project
    si.add_record(p.id, "landuse", {"title": "Sensitive", "details": "do not share"}, db)
    si.add_record(p.id, "activity", {"title": "Open house"}, db)
    records = si.list_records(p.id, db)
    hidden = si.redact(records, show_restricted=False)
    assert "do not share" not in hidden.to_csv()
    assert "do not share" in si.redact(records, show_restricted=True).to_csv()
    exported = si.export_view(records)
    assert list(exported["title"]) == ["Open house"]


def test_summary_and_overdue(db_project):
    db, p = db_project
    si.add_record(p.id, "commitment", {"title": "Fund monitoring", "due_date": "2026-01-01"}, db)
    si.add_record(p.id, "commitment", {"title": "Done", "due_date": "2026-01-01", "status": "Completed"}, db)
    si.add_record(p.id, "concern", {"title": "Noise", "status": "Unresolved (disagreement remains)"}, db)
    s = si.summarise(si.list_records(p.id, db), today=date(2026, 6, 1))
    assert s["overdue"] == 1 and s["open_items"] == 1
    assert s["unresolved"] == 1 and s["open_concerns"] == 1


def test_route_status_is_informational_text_only(db_project):
    db, p = db_project
    assert si.route_social_status(si.list_records(p.id, db), "direct", "Not assessed") == "Not assessed"
    si.add_record(p.id, "activity", {"title": "Meeting"}, db)
    text = si.route_social_status(si.list_records(p.id, db), "direct", "Not assessed")
    assert "not evidence of no concern" in text and "Not scored" in text
    si.add_record(p.id, "concern", {"title": "Water", "linked_route": "direct"}, db)
    text = si.route_social_status(si.list_records(p.id, db), "direct", "Not assessed")
    assert text.startswith("1 open concern")
    assert "consent" not in text.lower()


def test_records_deleted_with_project(db_project):
    db, p = db_project
    si.add_record(p.id, "activity", {"title": "Meeting"}, db)
    delete_project(p.id, db)
    assert si.list_records(p.id, db).empty


def test_overdue_with_no_due_dates(db_project):
    db, p = db_project
    si.add_record(p.id, "commitment", {"title": "No date yet"}, db)
    assert si.summarise(si.list_records(p.id, db))["overdue"] == 0
