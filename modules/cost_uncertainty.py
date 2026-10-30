"""Probabilistic cost estimates (Monte Carlo simulation).

The single-number estimate uses "most likely" values plus a flat contingency %.
Here every uncertain input instead gets a range (low / most likely / high, as a %
of the most likely value) and the estimate is recalculated thousands of times with
values drawn from triangular distributions. The spread of results gives:

* P10 / P50 / P80 / P90 costs: e.g. P80 = 80% of simulated outcomes cost this or less.
* The probability of staying within the budget.
* Implied contingency: how much must be added to the most-likely estimate to reach
  P50 / P80 / P90 confidence, compared with the contingency % currently assumed.

Contingency itself is NOT simulated: the simulation is what tells you how much is needed.

Limits (stated in the app): inputs are drawn independently, which tends to understate
the spread (in reality, e.g. labour and materials prices often rise together); ranges
are illustrative defaults; risks outside the listed inputs (scope changes, delays,
regulatory outcomes) are not included.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from types import SimpleNamespace

import numpy as np
import pandas as pd

from modules.cost_engine import CostAssumptions, cost_amounts, route_quantities

PERCENTILES = [10, 50, 80, 90]

# Default ranges: (low %, high %) relative to the most likely value. Skewed upwards because
# early-stage pipeline costs more often overrun than underrun.
DEFAULT_ASSUMPTION_RANGES: dict[str, tuple[float, float]] = {
    "steel_price_cad_per_t": (-20, 30), "wall_thickness_mm": (-5, 10), "coating_cad_per_m": (-20, 30),
    "construction_cad_per_inch_km": (-15, 35), "terrain_multiplier_per_point": (-30, 50),
    "water_crossing_cad_per_inch": (-30, 60), "wetland_premium_cad_per_km": (-30, 60),
    "forest_clearing_cad_per_km": (-20, 30), "row_width_m": (-10, 20), "land_cost_cad_per_ha": (-30, 60),
    "shared_corridor_land_discount_pct": (-40, 20), "access_road_cad_per_km": (-20, 40),
    "engineering_permitting_pct": (-20, 30), "environmental_mitigation_pct": (-20, 50),
    "restoration_cad_per_km": (-30, 50), "om_pct_of_capex": (-20, 40), "integrity_cad_per_km_yr": (-20, 40),
    "discount_rate_pct": (-25, 25), "decommissioning_pct_of_capex": (-40, 60),
}
QUANTITY_LABELS = {
    "length_km": "Route length (final alignment)", "water_crossings": "Number of water crossings",
    "wetland_km": "Wetland length crossed", "forest_clearing_km": "Forest clearing length",
    "new_access_km": "New access roads",
}
DEFAULT_QUANTITY_RANGES: dict[str, tuple[float, float]] = {
    "length_km": (-2, 8), "water_crossings": (-20, 30), "wetland_km": (-30, 50),
    "forest_clearing_km": (-20, 30), "new_access_km": (-30, 50),
}


def default_ranges() -> dict[str, tuple[float, float]]:
    return {**DEFAULT_ASSUMPTION_RANGES, **DEFAULT_QUANTITY_RANGES}


def range_labels() -> dict[str, str]:
    labels = {f.name: f.metadata["label"] for f in fields(CostAssumptions)}
    return {k: (QUANTITY_LABELS.get(k) or labels[k]) for k in default_ranges()}


def ranges_with_overrides(overrides: Mapping | None) -> dict[str, tuple[float, float]]:
    """Defaults, with any user-edited ranges applied (unknown keys ignored, low <= 0 <= high enforced)."""
    out = default_ranges()
    for k, v in (overrides or {}).items():
        if k in out:
            low, high = float(v[0]), float(v[1])
            out[k] = (min(low, 0.0), max(high, 0.0))
    return out


@dataclass
class MCResult:
    samples: pd.DataFrame  # capex_base, lifecycle_pv_base (CAD), one row per simulation
    most_likely_base: float  # deterministic estimate without contingency
    deterministic_capex: float  # deterministic estimate with the assumed contingency
    assumed_contingency_pct: float
    percentiles: pd.DataFrame  # P-level -> capex_base, lifecycle_pv_base, implied_contingency_pct

    def probability_within(self, budget_cad: float) -> float:
        return float((self.samples["capex_base"] <= budget_cad).mean())

    @property
    def confidence_of_assumed_contingency(self) -> float:
        """Share of simulated outcomes covered by most-likely estimate + assumed contingency."""
        return self.probability_within(self.deterministic_capex)


def _triangular(rng, mode, low_pct, high_pct, n):
    low, high = mode * (1 + low_pct / 100), mode * (1 + high_pct / 100)
    if np.isclose(low, high):
        return np.full(n, float(mode))
    return rng.triangular(min(low, high), mode, max(low, high), n)


def simulate(route: Mapping, diameter_mm: float, a: CostAssumptions, ranges: Mapping[str, tuple[float, float]],
             n: int = 5000, seed: int = 1) -> MCResult:
    rng = np.random.default_rng(seed)
    base_a = {f.name: getattr(a, f.name) for f in fields(CostAssumptions)}
    params = {k: (_triangular(rng, v, *ranges[k], n) if k in ranges else v) for k, v in base_a.items()}
    params["contingency_pct"] = 0.0
    q = route_quantities(route)
    quantities = {k: (_triangular(rng, v, *ranges[k], n) if k in ranges else v) for k, v in q.items()}
    res = cost_amounts(quantities, diameter_mm, SimpleNamespace(**params))
    samples = pd.DataFrame({"capex_base": np.broadcast_to(res["capex"], n),
                            "lifecycle_pv_base": np.broadcast_to(res["lifecycle_pv"], n)})

    most_likely = float(cost_amounts(q, diameter_mm, SimpleNamespace(**{**base_a, "contingency_pct": 0.0}))["capex"])
    deterministic = float(cost_amounts(q, diameter_mm, a)["capex"])
    pct = pd.DataFrame({
        "level": [f"P{p}" for p in PERCENTILES],
        "capex_base": [float(np.percentile(samples["capex_base"], p)) for p in PERCENTILES],
        "lifecycle_pv_base": [float(np.percentile(samples["lifecycle_pv_base"], p)) for p in PERCENTILES],
    })
    pct["implied_contingency_pct"] = 100 * (pct["capex_base"] / most_likely - 1)
    return MCResult(samples, most_likely, deterministic, a.contingency_pct, pct)
