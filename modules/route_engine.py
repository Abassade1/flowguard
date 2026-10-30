"""Multi-criteria route comparison and plain-language trade-off explanations.

Design principles (these matter more than the code):

* No single "winner" is chosen automatically. Routes are scored against several
  weighted criteria, and the trade-offs are explained in words.
* Hard constraints (e.g. crossing a protected area) are NOT folded into the
  score. A route with an unresolved hard constraint is flagged and left unranked,
  however well it scores on other criteria.
* Social, Indigenous and cultural considerations are NOT scored or monetised.
  They are shown as "not assessed" until authorised information comes from
  engagement with rights-holders and communities (see Module E, Phase 4).
* Unknowns are listed explicitly. Missing data is never treated as "no impact".

All route attributes below are SYNTHETIC demonstration values.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from modules.config import SOCIAL_NOT_ASSESSED
from modules.cost_engine import CostAssumptions, add_costs


@dataclass(frozen=True)
class Criterion:
    key: str
    label: str
    unit: str
    category: str  # "Cost", "Engineering" or "Environmental"
    lower_is_better: bool
    default_weight: float


CRITERIA: list[Criterion] = [
    Criterion("capex_cad", "CAPEX (estimated)", "CAD", "Cost", True, 2.0),
    Criterion("lifecycle_pv_cad", "Lifecycle cost (PV)", "CAD", "Cost", True, 1.0),
    Criterion("length_km", "Route length", "km", "Engineering", True, 1.0),
    Criterion("terrain_difficulty", "Terrain difficulty", "1-5", "Engineering", True, 1.5),
    Criterion("max_slope_pct", "Maximum slope", "%", "Engineering", True, 1.0),
    Criterion("new_access_km", "New access roads", "km", "Engineering", True, 1.0),
    Criterion("water_crossings", "Water crossings", "count", "Environmental", True, 2.0),
    Criterion("wetland_km", "Wetland crossing", "km", "Environmental", True, 2.0),
    Criterion("forest_clearing_km", "Forest clearing", "km", "Environmental", True, 1.5),
    Criterion("shared_corridor_km", "Shared existing corridor", "km", "Environmental", False, 1.5),
]

CRITERIA_BY_KEY = {c.key: c for c in CRITERIA}


def synthetic_route_alternatives() -> pd.DataFrame:
    """Four fixed, clearly labelled synthetic alternatives (used by tests and as a simple example).

    The app itself generates alternatives from geography (see modules/routing.py).
    """
    routes = pd.DataFrame(
        [
            {
                "route_id": "A", "name": "A: Direct corridor",
                "description": "Shortest alignment; cuts across upland and a protected area.",
                "length_km": 118.0, "terrain_difficulty": 3.4, "max_slope_pct": 18.0,
                "water_crossings": 6, "wetland_km": 4.2, "forest_clearing_km": 31.0,
                "new_access_km": 22.0, "shared_corridor_km": 10.0,
                "hard_constraints": "Crosses synthetic protected area PA-1 (approx. 6 km)",
                "geohazards": "Steep slope segment near KP 40-46 (synthetic); landslide susceptibility not assessed",
            },
            {
                "route_id": "B", "name": "B: Northern bypass",
                "description": "Avoids the protected area and main wetland by routing north through forest.",
                "length_km": 134.0, "terrain_difficulty": 2.8, "max_slope_pct": 12.0,
                "water_crossings": 4, "wetland_km": 0.8, "forest_clearing_km": 44.0,
                "new_access_km": 35.0, "shared_corridor_km": 25.0,
                "hard_constraints": "",
                "geohazards": "Discontinuous permafrost possible in north (synthetic, unverified)",
            },
            {
                "route_id": "C", "name": "C: Southern shared corridor",
                "description": "Follows an existing highway/utility right-of-way for most of its length.",
                "length_km": 141.0, "terrain_difficulty": 2.2, "max_slope_pct": 9.0,
                "water_crossings": 7, "wetland_km": 2.5, "forest_clearing_km": 12.0,
                "new_access_km": 8.0, "shared_corridor_km": 96.0,
                "hard_constraints": "",
                "geohazards": "Several crossings of existing buried utilities (synthetic)",
            },
            {
                "route_id": "D", "name": "D: River valley",
                "description": "Gentle grades along a valley floor; many stream and wetland crossings.",
                "length_km": 126.0, "terrain_difficulty": 2.5, "max_slope_pct": 6.0,
                "water_crossings": 11, "wetland_km": 6.8, "forest_clearing_km": 18.0,
                "new_access_km": 15.0, "shared_corridor_km": 40.0,
                "hard_constraints": "",
                "geohazards": "Flood exposure and river-bank erosion likely (synthetic); scour assessment needed",
            },
        ]
    )
    routes["social_indigenous"] = SOCIAL_NOT_ASSESSED
    routes["cultural_heritage"] = "Not assessed: authorised information required"
    routes["data_quality"] = "Synthetic demonstration"
    return add_costs(routes, diameter_mm=610.0, a=CostAssumptions())


def score_routes(routes: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Weighted multi-criteria score (0-100, relative to the alternatives shown).

    Each criterion is rescaled so the best alternative gets 1 and the worst gets 0,
    then combined using the weights. A score of 100 would mean "best on every
    weighted criterion"; it is NOT an absolute measure of acceptability.
    """
    weights = weights or {c.key: c.default_weight for c in CRITERIA}
    out = routes.copy()
    total_weight = sum(max(weights.get(c.key, 0.0), 0.0) for c in CRITERIA)
    weighted = np.zeros(len(out))
    for c in CRITERIA:
        values = out[c.key].astype(float).to_numpy()
        span = values.max() - values.min()
        if span == 0:
            norm = np.ones_like(values)
        elif c.lower_is_better:
            norm = (values.max() - values) / span
        else:
            norm = (values - values.min()) / span
        out[f"norm_{c.key}"] = norm
        weighted += max(weights.get(c.key, 0.0), 0.0) * norm
    out["score"] = 100 * weighted / total_weight if total_weight > 0 else 0.0
    out["has_hard_constraint"] = out["hard_constraints"].fillna("").str.len() > 0

    # Rank only routes without unresolved hard constraints.
    eligible = out.loc[~out["has_hard_constraint"], "score"]
    out["rank"] = pd.NA
    out.loc[eligible.index, "rank"] = eligible.rank(ascending=False, method="min").astype(int)
    out["comparison_status"] = np.where(
        out["has_hard_constraint"], "Unresolved hard constraint: not ranked", "Comparable"
    )
    return out


def explain_tradeoffs(scored: pd.DataFrame) -> dict[str, list[str]]:
    """Plain-language strengths, weaknesses, constraints and unknowns for each route."""
    notes: dict[str, list[str]] = {}
    for _, r in scored.iterrows():
        lines = []
        best = [c.label for c in CRITERIA if r[f"norm_{c.key}"] >= 0.999]
        worst = [c.label for c in CRITERIA if r[f"norm_{c.key}"] <= 0.001]
        if r["has_hard_constraint"]:
            lines.append(f"⛔ Hard constraint: {r['hard_constraints']}. Must be resolved or avoided; "
                         "it cannot be offset by lower cost.")
        if best:
            lines.append("✅ Best of the alternatives on: " + ", ".join(best) + ".")
        if worst:
            lines.append("⚠️ Weakest of the alternatives on: " + ", ".join(worst) + ".")
        if r["geohazards"]:
            lines.append(f"🪨 Geohazards / construction challenges: {r['geohazards']}.")
        if str(r["social_indigenous"]).startswith("Not assessed"):
            lines.append(f"❓ Unknown: social & Indigenous considerations ({r['social_indigenous']}); "
                         f"cultural heritage ({r['cultural_heritage']}).")
        else:
            lines.append(f"🤝 Engagement (information, not scored): {r['social_indigenous']}")
            lines.append(f"❓ Unknown: cultural heritage ({r['cultural_heritage']}).")
        notes[r["route_id"]] = lines
    return notes
