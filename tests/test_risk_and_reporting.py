"""Tests for the risk register and investor reports."""

import io
import zipfile

import pandas as pd
import pytest

from modules import risk_register as rr
from modules import social_impact as si
from modules.config import SOCIAL_NOT_ASSESSED
from modules.cost_engine import CostAssumptions, add_costs, assumption_table, estimate_route_cost, sensitivity
from modules.elevation import SyntheticElevation, register_entry
from modules.environmental_engine import screen_routes
from modules.gis_layers import assemble_layers, corridor_frame
from modules.projects import load_sample_project, save_project
from modules.reporting import DATA_CLASSES, _latin1, build_report, to_csv_zip, to_pdf
from modules.route_engine import score_routes
from modules.routing import STRATEGIES, generate_alternatives


def test_scoring_levels():
    assert rr.rating(5, 5) == 25 and rr.level(25) == "Very high"
    assert rr.level(4) == "Low" and rr.level(5) == "Medium" and rr.level(10) == "High" and rr.level(15) == "Very high"


def test_risk_crud_and_matrix(tmp_path):
    db = tmp_path / "t.db"
    p = save_project(load_sample_project(), db)
    rid = rr.add_risk(p.id, {"title": "Wetland crossing delays", "category": "Environmental", "likelihood": 4,
                             "consequence": 3, "residual_likelihood": 2, "residual_consequence": 3}, db)
    rr.add_risk(p.id, {"title": "Closed one", "category": "Engineering", "likelihood": 5, "consequence": 5,
                       "status": "Closed"}, db)
    risks = rr.list_risks(p.id, db)
    r = risks.set_index("id").loc[rid]
    assert (r["score"], r["level"], r["residual_score"], r["residual_level"]) == (12, "High", 6, "Medium")
    grid = rr.matrix_counts(risks)
    assert grid.loc[4, 3] == 1 and grid.to_numpy().sum() == 1  # closed risks excluded
    assert rr.matrix_counts(risks, residual=True).loc[2, 3] == 1
    rr.update_risk(rid, {"status": "Monitoring", "mitigation": "Winter construction"}, db)
    assert rr.list_risks(p.id, db).set_index("id").loc[rid, "status"] == "Monitoring"
    rr.delete_risk(rid, db)
    assert len(rr.list_risks(p.id, db)) == 1
    with pytest.raises(ValueError):
        rr.add_risk(p.id, {"title": "x", "category": "Engineering", "likelihood": 6, "consequence": 1}, db)
    with pytest.raises(ValueError):
        rr.add_risk(p.id, {"title": " ", "category": "Engineering", "likelihood": 1, "consequence": 1}, db)


@pytest.fixture(scope="module")
def analysis():
    p = load_sample_project()
    f = corridor_frame(p)
    layers, register = assemble_layers(f, [])
    el = SyntheticElevation(f)
    register.append(register_entry(el))
    routes, geoms, _ = generate_alternatives(f, layers, [s.key for s in STRATEGIES], elevation=el)
    a = CostAssumptions()
    routes = add_costs(routes, p.diameter_mm, a)
    routes["social_indigenous"] = SOCIAL_NOT_ASSESSED
    scored = score_routes(routes)
    est = {r["route_id"]: estimate_route_cost(r, p.diameter_mm, a) for _, r in scored.iterrows()}
    sens = sensitivity(scored.set_index("route_id").loc["R1"], p.diameter_mm, a)
    screening = screen_routes(f, scored, geoms, layers, elevation=el)
    return p, scored, est, a, sens, screening, register


def test_suggestions_cover_modules_and_frame_social_risks_correctly(analysis):
    p, scored, est, a, sens, screening, register = analysis
    s = rr.suggest_risks(scored, screening, "direct", {"activities": 0}, 1.0,
                         dict(zip(scored["strategy"], scored["capex_cad"])), sens["assumption"].iloc[0], register)
    cats = {x.category for x in s}
    assert {"Environmental", "Cost & schedule", "Data & uncertainty", "Social & community"} <= cats
    assert any(x.key == "budget:direct" for x in s)  # tiny budget -> overrun suggested
    social = next(x for x in s if x.key == "social:none")
    assert "cannot be offset" in social.description
    assert len({x.key for x in s}) == len(s)  # stable, unique keys


def test_accepted_suggestions_are_not_offered_again(tmp_path):
    db = tmp_path / "t.db"
    p = save_project(load_sample_project(), db)
    sug = rr.Suggestion("gap:Test", "Data gap: Test", "Data & uncertainty", "d", None, "Unit test")
    rr.add_risk(p.id, {"title": sug.title, "category": sug.category, "likelihood": 2, "consequence": 2,
                       "source": rr.suggestion_source(sug)}, db)
    assert "gap:Test" in rr.accepted_keys(rr.list_risks(p.id, db))


def test_report_sections_classes_and_exports(analysis, tmp_path):
    p, scored, est, a, sens, screening, register = analysis
    db = tmp_path / "t.db"
    saved = save_project(p, db)
    si.add_record(saved.id, "commitment", {"title": "Fund water monitoring", "category": "Mitigation commitment"}, db)
    si.add_record(saved.id, "commitment", {"title": "SECRET restricted promise", "sensitivity": "Restricted"}, db)
    records = si.list_records(saved.id, db)
    commitments = si.export_view(records[records["record_type"] == "commitment"])
    rr.add_risk(saved.id, {"title": "Wetland crossing delays", "category": "Environmental", "likelihood": 4,
                           "consequence": 3, "mitigation": "Winter construction"}, db)
    report = build_report(p, scored, "R1", est, assumption_table(a), sens, screening, register,
                          si.summarise(records), commitments, rr.list_risks(saved.id, db))
    titles = [s.title for s in report.sections]
    assert len(titles) == 9 and titles[0].startswith("1. Status")
    for s in report.sections:
        assert s.classes and set(s.classes) <= set(DATA_CLASSES)
    assert "SYNTHETIC" in report.sections[2].classes  # routes rest on synthetic layers here

    pdf = to_pdf(report)
    assert pdf[:4] == b"%PDF" and len(pdf) > 5000
    z = zipfile.ZipFile(io.BytesIO(to_csv_zip(report)))
    names = set(z.namelist())
    assert {"README.txt", "routes.csv", "costs.csv", "risks.csv", "mitigation_commitments.csv"} <= names
    everything = "".join(z.read(n).decode() for n in names)
    assert "SECRET restricted promise" not in everything
    assert "Fund water monitoring" in everything
    assert "data_class" in z.read("routes.csv").decode().splitlines()[0]
    costs = pd.read_csv(z.open("costs.csv"))
    assert (costs["data_class"] == "ESTIMATE").all()


def test_latin1_sanitising():
    assert _latin1("−5 ≤ x → y ✅") == "-5 <= x -> y"
    assert _latin1("10³ m³ · 25 °C") == "10³ m³ · 25 °C"
