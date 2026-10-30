"""Configurable pipeline cost estimation: CAPEX, OPEX, lifecycle cost and sensitivity.

Every number comes from an editable assumption in `CostAssumptions`. The default
values are ILLUSTRATIVE (round, generic figures for demonstration). They are not
benchmarks for any real project and must be replaced with project-specific
engineering and commercial data before any investment use.

All amounts are calculated in CAD. Pages convert to the project's currency for display.

How CAPEX is built
------------------
direct costs   = materials + construction + crossings & wet ground + land, access & clearing
allowances     = engineering & permitting % and environmental mitigation % of direct costs,
                 plus restoration per km of wetland/forest disturbed
contingency    = contingency % of (direct costs + allowances)
CAPEX          = direct costs + allowances + contingency

OPEX (per year) = O&M % of CAPEX + integrity/monitoring per km + energy (user-entered)
Lifecycle       = CAPEX + OPEX over the design life + decommissioning
                  (shown both undiscounted and as present value at the discount rate)
"""

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace

import numpy as np
import pandas as pd

STEEL_DENSITY_T_PER_M3 = 7.85
MM_PER_INCH = 25.4

COST_GROUPS = [
    "Materials",
    "Construction",
    "Crossings & wet ground",
    "Land, access & clearing",
    "Engineering & environmental allowances",
    "Contingency",
]

ASSUMPTION_GROUPS = ["Materials", "Construction", "Land & access", "Allowances & contingency",
                     "Operations & lifecycle"]


def _a(default, label, unit, group, help_text="", step=None, sensitivity=True):
    """Shorthand for an assumption field with display metadata."""
    return field(default=default, metadata={
        "label": label, "unit": unit, "group": group, "help": help_text,
        "step": step if step is not None else (default / 10 if default else 1.0), "sensitivity": sensitivity,
    })


@dataclass(frozen=True)
class CostAssumptions:
    # Materials
    steel_price_cad_per_t: float = _a(1800.0, "Line pipe steel price", "CAD/t", "Materials", step=50.0)
    wall_thickness_mm: float = _a(12.7, "Pipe wall thickness", "mm", "Materials",
                                  "Sets steel tonnage. Real values come from pressure design.", step=0.1)
    coating_cad_per_m: float = _a(60.0, "Coating & corrosion protection", "CAD/m", "Materials", step=5.0)
    # Construction
    construction_cad_per_inch_km: float = _a(
        55_000.0, "Mainline construction rate", "CAD per inch of diameter per km", "Construction",
        "Labour, equipment, welding, trenching, backfill for normal terrain.", step=1000.0)
    terrain_multiplier_per_point: float = _a(
        0.15, "Terrain multiplier per difficulty point", "fraction", "Construction",
        "Construction cost rises by this fraction for each terrain-difficulty point above 1 (scale 1-5).", step=0.01)
    water_crossing_cad_per_inch: float = _a(
        50_000.0, "Water crossing cost", "CAD per inch of diameter per crossing", "Construction",
        "Trenchless (e.g. HDD) or isolated open-cut crossing, averaged.", step=5000.0)
    wetland_premium_cad_per_km: float = _a(
        2_000_000.0, "Wetland construction premium", "CAD/km", "Construction",
        "Extra cost for matting, special methods and timing windows in wet ground.", step=100_000.0)
    forest_clearing_cad_per_km: float = _a(60_000.0, "Forest clearing", "CAD/km", "Construction", step=5000.0)
    # Land & access
    row_width_m: float = _a(30.0, "Right-of-way width", "m", "Land & access", step=1.0)
    land_cost_cad_per_ha: float = _a(15_000.0, "Land rights cost", "CAD/ha", "Land & access",
                                     "Easement / acquisition cost per hectare of right-of-way.", step=1000.0)
    shared_corridor_land_discount_pct: float = _a(
        50.0, "Land discount in shared corridors", "%", "Land & access",
        "Lower land cost where the route parallels an existing corridor.", step=5.0)
    access_road_cad_per_km: float = _a(400_000.0, "New access road cost", "CAD/km", "Land & access", step=20_000.0)
    # Allowances & contingency
    engineering_permitting_pct: float = _a(10.0, "Engineering & permitting", "% of direct cost",
                                           "Allowances & contingency", step=0.5)
    environmental_mitigation_pct: float = _a(3.0, "Environmental mitigation allowance", "% of direct cost",
                                             "Allowances & contingency", step=0.5)
    restoration_cad_per_km: float = _a(
        150_000.0, "Restoration of disturbed wetland/forest", "CAD/km", "Allowances & contingency", step=10_000.0)
    contingency_pct: float = _a(20.0, "Contingency", "% of direct + allowances", "Allowances & contingency",
                                "Covers unknowns at this early stage; typically falls as design matures.", step=1.0)
    # Operations & lifecycle
    om_pct_of_capex: float = _a(1.5, "Operations & maintenance", "% of CAPEX per year", "Operations & lifecycle",
                                step=0.1)
    integrity_cad_per_km_yr: float = _a(5_000.0, "Integrity & monitoring", "CAD per km per year",
                                        "Operations & lifecycle", "Inspection, ROW patrol, monitoring.", step=500.0)
    energy_cad_per_yr: float = _a(
        0.0, "Pumping / compression energy", "CAD per year", "Operations & lifecycle",
        "NOT estimated by this tool (depends on hydraulics). 0 means excluded; enter a figure from a "
        "hydraulic study.", step=100_000.0, sensitivity=False)
    design_life_years: float = _a(30.0, "Design life", "years", "Operations & lifecycle", step=1.0)
    discount_rate_pct: float = _a(6.0, "Discount rate", "% per year", "Operations & lifecycle", step=0.5)
    decommissioning_pct_of_capex: float = _a(5.0, "Decommissioning", "% of CAPEX at end of life",
                                             "Operations & lifecycle", step=0.5)

    @classmethod
    def from_overrides(cls, overrides: Mapping | None) -> "CostAssumptions":
        """Defaults, with any user-edited values applied (unknown keys are ignored)."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in (overrides or {}).items() if k in known})

    def overrides(self) -> dict:
        """Only the values that differ from the defaults (what gets saved with a project)."""
        default = CostAssumptions()
        return {k: v for k, v in asdict(self).items() if v != getattr(default, k)}


def assumption_table(assumptions: CostAssumptions) -> pd.DataFrame:
    """Register of every assumption, its value, default and status."""
    default = CostAssumptions()
    rows = []
    for f in fields(CostAssumptions):
        value, base = getattr(assumptions, f.name), getattr(default, f.name)
        rows.append({"group": f.metadata["group"], "assumption": f.metadata["label"], "value": value,
                     "unit": f.metadata["unit"], "default": base,
                     "status": "User-edited" if value != base else "Illustrative default", "key": f.name})
    return pd.DataFrame(rows)


@dataclass
class CostEstimate:
    lines: pd.DataFrame  # group, item, basis, amount_cad
    capex: float
    opex_annual: float
    lifecycle_nominal: float
    lifecycle_pv: float
    length_km: float
    notes: list[str]

    @property
    def capex_per_km(self) -> float:
        return self.capex / self.length_km if self.length_km else math.nan

    def by_group(self) -> pd.Series:
        return self.lines.groupby("group", sort=False)["amount_cad"].sum().reindex(COST_GROUPS, fill_value=0.0)


def steel_tonnes_per_km(diameter_mm: float, wall_mm: float) -> float:
    """Steel mass of the pipe wall: π × (D − t) × t × density."""
    d, t = diameter_mm / 1000, wall_mm / 1000
    return math.pi * (d - t) * t * 1000 * STEEL_DENSITY_T_PER_M3


def annuity_factor(rate, years):
    """Present value of 1 per year for `years` years at `rate` (0.06 = 6%). Works on arrays too."""
    rate = np.asarray(rate, dtype=float)
    safe = np.where(rate == 0, 1.0, rate)
    out = np.where(rate == 0, years, (1 - (1 + safe) ** -np.asarray(years, dtype=float)) / safe)
    return float(out) if out.ndim == 0 else out


# Cost lines in report order: (group, item).
LINE_ITEMS = [
    ("Materials", "Line pipe steel"), ("Materials", "Coating & corrosion protection"),
    ("Construction", "Mainline construction"),
    ("Crossings & wet ground", "Water crossings"), ("Crossings & wet ground", "Wetland construction premium"),
    ("Land, access & clearing", "Right-of-way land rights"), ("Land, access & clearing", "New access roads"),
    ("Land, access & clearing", "Forest clearing"),
    ("Engineering & environmental allowances", "Engineering & permitting"),
    ("Engineering & environmental allowances", "Environmental mitigation allowance"),
    ("Engineering & environmental allowances", "Restoration of disturbed wetland/forest"),
    ("Contingency", "Contingency"),
]


def route_quantities(route: Mapping) -> dict:
    """The measured route quantities the cost model uses."""
    return {"length_km": float(route["length_km"]), "terrain_difficulty": float(route["terrain_difficulty"]),
            "water_crossings": float(route["water_crossings"]), "wetland_km": float(route["wetland_km"]),
            "forest_clearing_km": float(route["forest_clearing_km"]), "new_access_km": float(route["new_access_km"]),
            "shared_corridor_km": float(route.get("shared_corridor_km", 0.0))}


def cost_amounts(q: Mapping, diameter_mm: float, a) -> dict:
    """The cost arithmetic, on its own.

    `q` holds route quantities and `a` the assumptions (a CostAssumptions, or any object with the
    same attribute names). Every value may be a number or a NumPy array of equal length, so the
    same formulas serve both the single estimate and the Monte Carlo simulation.
    """
    L = q["length_km"]
    inches = diameter_mm / MM_PER_INCH
    item = {}
    item["Line pipe steel"] = steel_tonnes_per_km(diameter_mm, a.wall_thickness_mm) * L * a.steel_price_cad_per_t
    item["Coating & corrosion protection"] = L * 1000 * a.coating_cad_per_m
    terrain_factor = 1 + a.terrain_multiplier_per_point * (q["terrain_difficulty"] - 1)
    item["Mainline construction"] = inches * L * a.construction_cad_per_inch_km * terrain_factor
    item["Water crossings"] = q["water_crossings"] * inches * a.water_crossing_cad_per_inch
    item["Wetland construction premium"] = q["wetland_km"] * a.wetland_premium_cad_per_km
    shared_share = np.where(np.asarray(L) > 0, np.minimum(q["shared_corridor_km"], L) / np.maximum(L, 1e-9), 0.0)
    hectares = L * a.row_width_m / 10  # km × m → ha (1 km × 10 m = 1 ha)
    item["Right-of-way land rights"] = (hectares * a.land_cost_cad_per_ha
                                        * (1 - shared_share * a.shared_corridor_land_discount_pct / 100))
    item["New access roads"] = q["new_access_km"] * a.access_road_cad_per_km
    item["Forest clearing"] = q["forest_clearing_km"] * a.forest_clearing_cad_per_km
    direct = sum(item.values())
    item["Engineering & permitting"] = direct * a.engineering_permitting_pct / 100
    item["Environmental mitigation allowance"] = direct * a.environmental_mitigation_pct / 100
    item["Restoration of disturbed wetland/forest"] = ((q["wetland_km"] + q["forest_clearing_km"])
                                                       * a.restoration_cad_per_km)
    before_contingency = sum(item.values())
    item["Contingency"] = before_contingency * a.contingency_pct / 100

    capex = before_contingency + item["Contingency"]
    opex = capex * a.om_pct_of_capex / 100 + L * a.integrity_cad_per_km_yr + a.energy_cad_per_yr
    years, rate = a.design_life_years, np.asarray(a.discount_rate_pct) / 100
    decom = capex * a.decommissioning_pct_of_capex / 100
    return {"items": item, "before_contingency": before_contingency, "capex": capex, "opex": opex,
            "lifecycle_nominal": capex + opex * years + decom,
            "lifecycle_pv": capex + opex * annuity_factor(rate, years) + decom / (1 + rate) ** years}


def estimate_route_cost(route: Mapping, diameter_mm: float, a: CostAssumptions) -> CostEstimate:
    """Cost estimate for one route, with a readable basis for every line."""
    q = route_quantities(route)
    res = cost_amounts(q, diameter_mm, a)
    L, inches = q["length_km"], diameter_mm / MM_PER_INCH
    wet, forest = q["wetland_km"], q["forest_clearing_km"]
    shared = min(q["shared_corridor_km"], L)
    terrain_factor = 1 + a.terrain_multiplier_per_point * (q["terrain_difficulty"] - 1)
    basis = {
        "Line pipe steel": f"{steel_tonnes_per_km(diameter_mm, a.wall_thickness_mm) * L:,.0f} t × "
                           f"{a.steel_price_cad_per_t:,.0f} CAD/t",
        "Coating & corrosion protection": f"{L:,.1f} km × {a.coating_cad_per_m:,.0f} CAD/m",
        "Mainline construction": f"{inches:.0f} in × {L:,.1f} km × {a.construction_cad_per_inch_km:,.0f} × "
                                 f"terrain factor {terrain_factor:.2f}",
        "Water crossings": f"{int(q['water_crossings'])} × {inches:.0f} in × {a.water_crossing_cad_per_inch:,.0f} CAD/in",
        "Wetland construction premium": f"{wet:,.1f} km × {a.wetland_premium_cad_per_km:,.0f}",
        "Right-of-way land rights": f"{L * a.row_width_m / 10:,.0f} ha × {a.land_cost_cad_per_ha:,.0f} CAD/ha "
                                    f"({shared:,.1f} km shared at −{a.shared_corridor_land_discount_pct:.0f}%)",
        "New access roads": f"{q['new_access_km']:,.1f} km × {a.access_road_cad_per_km:,.0f}",
        "Forest clearing": f"{forest:,.1f} km × {a.forest_clearing_cad_per_km:,.0f}",
        "Engineering & permitting": f"{a.engineering_permitting_pct:.1f}% of direct cost",
        "Environmental mitigation allowance": f"{a.environmental_mitigation_pct:.1f}% of direct cost",
        "Restoration of disturbed wetland/forest": f"{wet + forest:,.1f} km × {a.restoration_cad_per_km:,.0f}",
        "Contingency": f"{a.contingency_pct:.0f}% of direct + allowances",
    }
    lines = pd.DataFrame([{"group": g, "item": i, "basis": basis[i], "amount_cad": float(res["items"][i])}
                          for g, i in LINE_ITEMS])

    notes = ["Illustrative estimate based on editable default assumptions; not for investment decisions."]
    if a.energy_cad_per_yr == 0:
        notes.append("Pumping/compression energy is NOT included in OPEX (enter it from a hydraulic study).")
    notes.append("Excludes pump/compressor stations, terminals, financing costs, taxes and inflation.")
    return CostEstimate(lines, float(res["capex"]), float(res["opex"]), float(res["lifecycle_nominal"]),
                        float(res["lifecycle_pv"]), L, notes)


def add_costs(routes: pd.DataFrame, diameter_mm: float, a: CostAssumptions) -> pd.DataFrame:
    """Add CAPEX, OPEX and lifecycle columns (CAD) to a routes table."""
    out = routes.copy()
    estimates = [estimate_route_cost(r, diameter_mm, a) for _, r in out.iterrows()]
    out["capex_cad"] = [e.capex for e in estimates]
    out["capex_per_km_cad"] = [e.capex_per_km for e in estimates]
    out["opex_annual_cad"] = [e.opex_annual for e in estimates]
    out["lifecycle_nominal_cad"] = [e.lifecycle_nominal for e in estimates]
    out["lifecycle_pv_cad"] = [e.lifecycle_pv for e in estimates]
    return out


def cumulative_cost(estimate: CostEstimate, a: CostAssumptions, discounted: bool = False) -> pd.DataFrame:
    """Cumulative spend by year: CAPEX in year 0, OPEX every year, decommissioning at the end."""
    years = int(round(a.design_life_years))
    rate = a.discount_rate_pct / 100
    rows, total = [], estimate.capex
    rows.append({"year": 0, "cumulative_cad": total})
    for y in range(1, years + 1):
        spend = estimate.opex_annual + (estimate.capex * a.decommissioning_pct_of_capex / 100 if y == years else 0)
        total += spend / (1 + rate) ** y if discounted else spend
        rows.append({"year": y, "cumulative_cad": total})
    return pd.DataFrame(rows)


SENSITIVITY_METRICS = {"capex": "CAPEX", "lifecycle_pv": "Lifecycle cost (present value)"}


def sensitivity(route: Mapping, diameter_mm: float, a: CostAssumptions, swing_pct: float = 25.0,
                metric: str = "capex") -> pd.DataFrame:
    """One-at-a-time sensitivity: move each assumption down/up by `swing_pct` and record the result."""
    base = getattr(estimate_route_cost(route, diameter_mm, a), metric)
    rows = []
    for f in fields(CostAssumptions):
        if not f.metadata["sensitivity"]:
            continue
        value = getattr(a, f.name)
        if value == 0:
            continue
        results = {}
        for side, factor in (("low", 1 - swing_pct / 100), ("high", 1 + swing_pct / 100)):
            results[side] = getattr(estimate_route_cost(route, diameter_mm, replace(a, **{f.name: value * factor})),
                                    metric)
        rows.append({
            "assumption": f.metadata["label"], "key": f.name,
            "low_value": value * (1 - swing_pct / 100), "high_value": value * (1 + swing_pct / 100),
            "result_low": results["low"], "result_high": results["high"],
            "change_low": results["low"] - base, "change_high": results["high"] - base,
        })
    df = pd.DataFrame(rows)
    df["swing"] = (df["result_high"] - df["result_low"]).abs()
    return df.sort_values("swing", ascending=False).reset_index(drop=True).assign(base=base)
