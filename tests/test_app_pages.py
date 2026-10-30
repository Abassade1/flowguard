"""Smoke tests: every page renders through the real app without raising an exception.

Uses Streamlit's built-in AppTest, which runs the app headlessly (no browser).
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "app_pages").glob("*.py"))


def open_page(page: str, **state) -> AppTest:
    """Start the app via app.py (as a user would), apply settings, then open a page."""
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    for key, value in state.items():
        at.session_state[key] = value
    return at.switch_page(page).run()


def test_main_app_runs():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert "Executive Overview" in at.title[0].value


@pytest.mark.parametrize("page", PAGES)
def test_page_runs(page):
    at = open_page(page)
    assert not at.exception, at.exception


@pytest.mark.parametrize("scenario", ["Flow meter fault (bias)", "Pressure sensor drift"])
def test_monitoring_with_ml_enabled(scenario):
    at = open_page("app_pages/7_Pipeline_Monitoring.py", pg_scenario=scenario, pg_ml_enabled=True)
    assert not at.exception, at.exception
    for model in ["isolation_forest", "robust_covariance"]:
        at.session_state["pg_ml_model"] = model
        at.run()
        assert not at.exception, at.exception


@pytest.mark.parametrize("mode", ["Planning", "Construction", "Operations"])
def test_twin_modes(mode):
    at = open_page("app_pages/6_Digital_Twin.py", pg_twin_mode=mode)
    assert not at.exception, at.exception


def test_invalid_project_shows_message_not_crash():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    bad = dict(at.session_state["pg_active_project"], end_lat=53.2, end_lon=-112.9)  # same as start
    at.session_state["pg_active_project"] = bad
    for page in ["app_pages/0_Executive_Overview.py", "app_pages/2_Route_Planning.py", "app_pages/6_Digital_Twin.py",
                 "app_pages/7_Pipeline_Monitoring.py"]:
        at = at.switch_page(page).run()
        assert not at.exception, (page, at.exception)


def test_monitoring_length_follows_modelled_route():
    at = open_page("app_pages/7_Pipeline_Monitoring.py")
    assert not at.exception
    assert at.session_state["pg_link_project"] is True


@pytest.mark.parametrize("scenario", ["Normal operations", "Pressure surge", "Sensor fault", "Temperature excursion"])
@pytest.mark.parametrize("page", ["app_pages/0_Executive_Overview.py", "app_pages/6_Digital_Twin.py",
                                  "app_pages/7_Pipeline_Monitoring.py"])
def test_operations_pages_handle_each_scenario(page, scenario):
    at = open_page(page, pg_scenario=scenario)
    assert not at.exception, at.exception


def test_new_project_page_and_route_options():
    at = open_page("app_pages/1_Project_Setup.py")
    at.button[1].click().run()  # "New project"
    assert not at.exception
    assert at.session_state["pg_active_project"]["name"] == "New pipeline project"
    at = at.switch_page("app_pages/2_Route_Planning.py").run()
    assert not at.exception
    at.toggle[0].set_value(False).run()  # allow constraint-crossing routes
    assert not at.exception


def test_cost_assumption_edit_marks_project_unsaved():
    at = open_page("app_pages/3_Cost_Optimizer.py")
    steel = next(n for n in at.number_input if n.key == "cost_steel_price_cad_per_t")
    steel.set_value(2500.0).run()
    assert not at.exception
    assert at.session_state["pg_active_project"]["cost_assumptions"] == {"steel_price_cad_per_t": 2500.0}
    at.radio[0].set_value("lifecycle_pv").run()  # sensitivity on lifecycle cost
    assert not at.exception


def test_engagement_page_add_record_with_saved_project():
    from modules import projects as store
    from modules import social_impact as si
    saved = store.save_project(store.load_sample_project())
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    at.session_state["pg_active_project"] = saved.to_dict()
    at = at.switch_page("app_pages/5_Community_Engagement.py").run()
    assert not at.exception
    title = next(t for t in at.text_input if t.key == "add_activity_title")
    title.input("Open house in the community hall")
    next(b for b in at.button if "Add activity" in str(b.label)).click().run()
    assert not at.exception
    assert list(si.list_records(saved.id)["title"]) == ["Open house in the community hall"]
    at = at.switch_page("app_pages/2_Route_Planning.py").run()  # social text now reflects engagement
    assert not at.exception


def test_risk_and_report_pages_with_saved_project():
    from modules import projects as store
    from modules import risk_register as rr
    saved = store.save_project(store.load_sample_project())
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90).run()
    at.session_state["pg_active_project"] = saved.to_dict()
    at = at.switch_page("app_pages/8_Risk_Register.py").run()
    assert not at.exception, at.exception
    add = next(b for b in at.button if b.label == "Add to register")  # accept the first suggestion
    add.click().run()
    assert not at.exception, at.exception
    assert len(rr.list_risks(saved.id)) == 1
    at = at.switch_page("app_pages/9_Investor_Reports.py").run()
    assert not at.exception, at.exception


def test_settings_survive_page_switch():
    at = open_page("app_pages/2_Route_Planning.py", pg_n_sections=12, pg_pipeline_type="Water")
    at = at.switch_page("app_pages/6_Digital_Twin.py").run()
    assert not at.exception
    assert at.session_state["pg_n_sections"] == 12
    assert at.session_state["pg_pipeline_type"] == "Water"
