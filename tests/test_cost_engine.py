"""Tests for the cost estimation engine."""

import math

import pytest

from modules.cost_engine import (
    COST_GROUPS, CostAssumptions, add_costs, annuity_factor, assumption_table, cumulative_cost,
    estimate_route_cost, sensitivity, steel_tonnes_per_km,
)
from modules.route_engine import synthetic_route_alternatives

ROUTE = {"length_km": 100.0, "terrain_difficulty": 2.0, "water_crossings": 4, "wetland_km": 3.0,
         "forest_clearing_km": 10.0, "new_access_km": 20.0, "shared_corridor_km": 40.0}


def test_steel_tonnage_formula():
    # 24-inch (610 mm) pipe, 12.7 mm wall: about 187 t per km.
    assert steel_tonnes_per_km(610, 12.7) == pytest.approx(187.1, abs=0.5)


def test_capex_adds_up_and_groups_are_complete():
    e = estimate_route_cost(ROUTE, 610, CostAssumptions())
    assert e.capex == pytest.approx(e.lines["amount_cad"].sum())
    assert list(e.by_group().index) == COST_GROUPS
    assert e.by_group().sum() == pytest.approx(e.capex)
    assert e.capex_per_km == pytest.approx(e.capex / 100)


def test_contingency_is_percentage_of_everything_before_it():
    a = CostAssumptions(contingency_pct=20.0)
    lines = estimate_route_cost(ROUTE, 610, a).lines
    before = lines[lines["group"] != "Contingency"]["amount_cad"].sum()
    assert lines[lines["group"] == "Contingency"]["amount_cad"].iloc[0] == pytest.approx(0.2 * before)


def test_costs_respond_to_drivers():
    a = CostAssumptions()
    base = estimate_route_cost(ROUTE, 610, a).capex
    assert estimate_route_cost({**ROUTE, "length_km": 150.0}, 610, a).capex > base
    assert estimate_route_cost({**ROUTE, "wetland_km": 10.0}, 610, a).capex > base
    assert estimate_route_cost({**ROUTE, "terrain_difficulty": 4.0}, 610, a).capex > base
    assert estimate_route_cost(ROUTE, 914, a).capex > base  # bigger pipe costs more
    # Sharing an existing corridor lowers land cost.
    assert estimate_route_cost({**ROUTE, "shared_corridor_km": 0.0}, 610, a).capex > base


def test_lifecycle_math():
    a = CostAssumptions(discount_rate_pct=0.0, decommissioning_pct_of_capex=0.0, design_life_years=10)
    e = estimate_route_cost(ROUTE, 610, a)
    assert e.lifecycle_pv == pytest.approx(e.capex + 10 * e.opex_annual)  # no discounting
    assert e.lifecycle_nominal == pytest.approx(e.lifecycle_pv)
    assert annuity_factor(0.06, 30) == pytest.approx(13.7648, rel=1e-4)
    discounted = estimate_route_cost(ROUTE, 610, CostAssumptions())
    assert discounted.lifecycle_pv < discounted.lifecycle_nominal


def test_energy_excluded_note_and_inclusion():
    e = estimate_route_cost(ROUTE, 610, CostAssumptions())
    assert any("NOT included" in n for n in e.notes)
    with_energy = estimate_route_cost(ROUTE, 610, CostAssumptions(energy_cad_per_yr=1e6))
    assert with_energy.opex_annual == pytest.approx(e.opex_annual + 1e6)


def test_overrides_round_trip():
    a = CostAssumptions(steel_price_cad_per_t=2000.0)
    assert a.overrides() == {"steel_price_cad_per_t": 2000.0}
    assert CostAssumptions.from_overrides({"steel_price_cad_per_t": 2000, "unknown": 1}) == a
    table = assumption_table(a)
    assert (table["status"] == "User-edited").sum() == 1


def test_sensitivity_is_sorted_and_symmetric_for_linear_inputs():
    sens = sensitivity(ROUTE, 610, CostAssumptions(), 25, "capex")
    assert sens["swing"].is_monotonic_decreasing
    assert sens.iloc[0]["key"] == "construction_cad_per_inch_km"
    row = sens.set_index("key").loc["steel_price_cad_per_t"]
    assert row["change_high"] == pytest.approx(-row["change_low"])
    assert "energy_cad_per_yr" not in set(sens["key"])
    pv = sensitivity(ROUTE, 610, CostAssumptions(), 25, "lifecycle_pv")
    assert "discount_rate_pct" in set(pv["key"])


def test_cumulative_cost_ends_at_lifecycle_totals():
    a = CostAssumptions()
    e = estimate_route_cost(ROUTE, 610, a)
    nominal = cumulative_cost(e, a)
    assert nominal["cumulative_cad"].iloc[0] == pytest.approx(e.capex)
    assert nominal["cumulative_cad"].iloc[-1] == pytest.approx(e.lifecycle_nominal)
    assert cumulative_cost(e, a, discounted=True)["cumulative_cad"].iloc[-1] == pytest.approx(e.lifecycle_pv)


def test_add_costs_columns():
    routes = add_costs(synthetic_route_alternatives(), 610, CostAssumptions())
    for col in ["capex_cad", "capex_per_km_cad", "opex_annual_cad", "lifecycle_pv_cad", "lifecycle_nominal_cad"]:
        assert routes[col].notna().all() and (routes[col] > 0).all()
    assert not math.isnan(routes["capex_per_km_cad"].iloc[0])
