"""Tests for the Monte Carlo cost simulation."""

import numpy as np
import pytest

from modules import cost_uncertainty as mc
from modules.cost_engine import CostAssumptions, cost_amounts, estimate_route_cost, route_quantities
from modules.risk_register import suggest_risks

ROUTE = {"length_km": 100.0, "terrain_difficulty": 2.0, "water_crossings": 4, "wetland_km": 3.0,
         "forest_clearing_km": 10.0, "new_access_km": 20.0, "shared_corridor_km": 40.0}


def test_vectorised_formulas_match_single_estimate():
    a = CostAssumptions()
    e = estimate_route_cost(ROUTE, 610, a)
    res = cost_amounts(route_quantities(ROUTE), 610, a)
    assert float(res["capex"]) == pytest.approx(e.capex)
    assert float(res["lifecycle_pv"]) == pytest.approx(e.lifecycle_pv)
    assert sum(float(v) for v in res["items"].values()) == pytest.approx(e.capex)


def test_zero_spread_reproduces_most_likely_estimate():
    a = CostAssumptions()
    zero = {k: (0.0, 0.0) for k in mc.default_ranges()}
    res = mc.simulate(ROUTE, 610, a, zero, n=500)
    assert np.ptp(res.samples["capex_base"]) == pytest.approx(0, abs=1e-6)
    assert res.samples["capex_base"].iloc[0] == pytest.approx(res.most_likely_base)
    # Without contingency, the deterministic estimate is base x (1 + contingency %).
    assert res.deterministic_capex == pytest.approx(res.most_likely_base * (1 + a.contingency_pct / 100))


def test_percentiles_ordered_and_implied_contingency_consistent():
    res = mc.simulate(ROUTE, 610, CostAssumptions(), mc.default_ranges(), n=4000, seed=3)
    p = res.percentiles.set_index("level")
    assert p["capex_base"].is_monotonic_increasing
    assert p.loc["P80", "implied_contingency_pct"] == pytest.approx(
        100 * (p.loc["P80", "capex_base"] / res.most_likely_base - 1))
    assert 0 < res.confidence_of_assumed_contingency <= 1
    assert res.probability_within(p.loc["P50", "capex_base"]) == pytest.approx(0.5, abs=0.02)


def test_reproducible_and_wider_ranges_widen_spread():
    a = CostAssumptions()
    r1 = mc.simulate(ROUTE, 610, a, mc.default_ranges(), n=2000, seed=5)
    r2 = mc.simulate(ROUTE, 610, a, mc.default_ranges(), n=2000, seed=5)
    assert r1.samples.equals(r2.samples)
    wide = {k: (2 * lo, 2 * hi) for k, (lo, hi) in mc.default_ranges().items()}
    r3 = mc.simulate(ROUTE, 610, a, wide, n=2000, seed=5)
    assert r3.samples["capex_base"].std() > r1.samples["capex_base"].std()


def test_overrides_are_applied_and_sanitised():
    ranges = mc.ranges_with_overrides({"steel_price_cad_per_t": [-50, 80], "length_km": [10, -5], "unknown": [1, 2]})
    assert ranges["steel_price_cad_per_t"] == (-50.0, 80.0)
    assert ranges["length_km"] == (0.0, 0.0)  # low must be <= 0 <= high
    assert "unknown" not in ranges
    assert set(mc.range_labels()) == set(mc.default_ranges())


def test_monte_carlo_risk_suggestions():
    import pandas as pd
    routes = pd.DataFrame([{"route_id": "R1", "strategy": "direct", "name": "R1: Direct", "hard_constraints": ""}])
    screening = pd.DataFrame(columns=["route_id", "theme", "label", "level", "explanation", "confidence"])
    keys = {s.key for s in suggest_risks(routes, screening, "direct", {"activities": 1}, 0, {}, None, [],
                                         contingency_check=(20.0, 25.0), budget_probability=0.3)}
    assert {"contingency:p80", "budget:probability"} <= keys
    keys = {s.key for s in suggest_risks(routes, screening, "direct", {"activities": 1}, 0, {}, None, [],
                                         contingency_check=(20.0, 15.0), budget_probability=0.9)}
    assert not keys & {"contingency:p80", "budget:probability"}
