"""Environmental screening support for route alternatives.

IMPORTANT: this is a screening aid for qualified professionals. It is NOT an
environmental assessment, and its outputs are not findings.

For each route and each environmental theme the engine reports:
* a measured indicator (from geometry, never guessed),
* a screening level: "High", "Moderate", "Low", or "Not assessed",
* where the data came from and how much confidence it deserves,
* standard mitigation ideas for professional review.

Rules that matter:
* If a theme's data is missing, the level is "Not assessed (data gap)", never "Low".
  Missing data is not evidence that there is no impact.
* Synthetic data gives a result with "None" confidence: useful for learning
  the tool, meaningless for decisions.
* Terrain-based themes (erosion, landslide, flood) are crude proxies, labelled as such.
"""

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString
from shapely.ops import unary_union

from modules.elevation import SyntheticElevation
from modules.gis_layers import CorridorFrame

LEVELS = ["High", "Moderate", "Low", "Not assessed"]
LEVEL_ORDER = {level: i for i, level in enumerate(LEVELS)}
LEVEL_COLORS = {"High": "#D64545", "Moderate": "#E0A030", "Low": "#2E9E6B", "Not assessed": "#8A8F98"}

ASSESSMENT_DISCLAIMER = (
    "Screening support only. This is not an environmental assessment and does not replace field "
    "studies, regulatory processes or qualified professional judgement. Levels are relative screening "
    "flags from simple rules. Missing data is shown as a data gap, never as 'no impact'."
)


@dataclass(frozen=True)
class EnvThresholds:
    """Screening distances and trigger values (illustrative, for demonstration)."""

    riparian_buffer_m: float = 30.0
    wetland_buffer_m: float = 100.0
    protected_buffer_m: float = 1000.0
    habitat_buffer_m: float = 500.0
    flood_proxy_buffer_m: float = 250.0
    erosion_grade_pct: float = 8.0
    erosion_near_water_m: float = 200.0
    landslide_grade_pct: float = 15.0
    row_width_m: float = 30.0
    access_road_width_m: float = 10.0


@dataclass(frozen=True)
class Theme:
    key: str
    label: str
    layers: tuple[str, ...]  # layer categories the theme depends on ("terrain" = elevation surface)
    mitigation: tuple[str, ...]


THEMES: list[Theme] = [
    Theme("wetlands", "Wetlands", ("wetland",), (
        "Avoid or minimise wetland crossings through route refinement.",
        "Consider trenchless crossing or construction in frozen/dry conditions.",
        "Use matting and maintain natural cross-drainage; restore wetland function after construction.")),
    Theme("water", "Rivers, lakes & riparian areas", ("water",), (
        "Assess trenchless (e.g. HDD) crossings where feasible; otherwise isolated crossing methods.",
        "Respect in-water timing windows for fish and aquatic life.",
        "Sediment and erosion control at approaches; watershed-level effects need separate assessment.")),
    Theme("protected", "Protected areas", ("protected",), (
        "Avoid. A crossing is a hard constraint unless formally resolved with the responsible authority.",
        "Maintain buffers from boundaries; confirm boundaries with official sources.")),
    Theme("forest", "Forest cover & clearing", ("forest",), (
        "Minimise right-of-way width through forest; reuse existing clearings.",
        "Avoid clearing during sensitive breeding/nesting periods.",
        "Reforestation and invasive-species management after construction.")),
    Theme("fragmentation", "Habitat fragmentation", ("road", "forest", "wetland"), (
        "Co-locate with existing linear corridors to limit new disturbance.",
        "Limit new access; control access after construction; consider wildlife crossings.",
        "Narrow the footprint through intact patches.")),
    Theme("habitat", "Wildlife habitat & biodiversity", ("habitat",), (
        "Obtain authorised habitat and species-at-risk data; commission field surveys by qualified biologists.",
        "Apply timing restrictions and setbacks identified by specialists.",
        "Biodiversity effects cannot be screened without data.")),
    Theme("erosion", "Soil erosion (proxy)", ("terrain", "water"), (
        "Prepare an erosion and sediment control plan; slope breakers and rapid revegetation.",
        "Salvage and replace topsoil; avoid wet-weather work on steep slopes near water.")),
    Theme("landslide", "Landslide susceptibility (proxy)", ("terrain",), (
        "Geotechnical investigation of steep segments; avoid unstable slopes.",
        "Slope stabilisation and ongoing monitoring where steep ground cannot be avoided.")),
    Theme("flood", "Flood exposure (proxy)", ("water", "terrain"), (
        "Hydrotechnical assessment of floodplains and crossings (scour, channel migration).",
        "Set burial depth below estimated scour; avoid floodplain routing where possible.")),
    Theme("disturbance", "Construction disturbance", (), (
        "Minimise footprint and temporary workspace; use existing access.",
        "Prepare a reclamation plan before construction.")),
    Theme("restoration", "Restoration requirements", ("forest", "wetland"), (
        "Reclamation plan with measurable success criteria and native species.",
        "Post-construction monitoring period agreed with regulators and communities.")),
]
THEMES_BY_KEY = {t.key: t for t in THEMES}


def layer_source(layers: dict[str, gpd.GeoDataFrame], category: str, terrain_synthetic: bool = True) -> str:
    """'user', 'synthetic' or 'missing' for a layer category ('terrain' = the elevation source)."""
    if category == "terrain":
        return "synthetic" if terrain_synthetic else "user"
    gdf = layers.get(category)
    if gdf is None or gdf.empty:
        return "missing" if category in ("habitat",) else "synthetic"
    return "synthetic" if gdf["synthetic"].all() else "user"


def confidence_for(theme: Theme, layers: dict[str, gpd.GeoDataFrame],
                   terrain_synthetic: bool = True) -> tuple[str, str]:
    """(data status, confidence) for a theme, driven by its weakest input."""
    sources = [layer_source(layers, c, terrain_synthetic) for c in theme.layers]
    if "missing" in sources:
        return "Data gap", "Not assessed"
    if not sources:
        return "Derived from route geometry", "Medium: depends on assumed widths"
    if "synthetic" in sources:
        proxy = "terrain" in theme.layers
        return ("Synthetic proxy" if proxy else "Synthetic demonstration"), "None: synthetic data"
    if "terrain" in theme.layers:
        return "User-supplied proxy", "Low: simple proxy from real terrain"
    return "User-supplied", "Medium: user data, coverage unverified"


def _union(gdf):
    return unary_union(list(gdf.geometry)) if gdf is not None and not gdf.empty else None


def _km(geom) -> float:
    return 0.0 if geom is None or geom.is_empty else geom.length / 1000.0


def _grade_profile(frame: CorridorFrame, route: LineString, elevation, step_m: float = 100.0):
    d = np.append(np.arange(0, route.length, step_m), route.length)
    pts = shapely.line_interpolate_point(route, d)
    elev = elevation.sample(shapely.get_x(pts), shapely.get_y(pts))
    grade = np.zeros_like(elev)
    grade[1:] = 100 * np.abs(np.diff(elev)) / np.maximum(np.diff(d), 1e-6)
    return pts, grade, step_m / 1000.0


def _level(value: float, moderate_above: float, high_above: float) -> str:
    if value > high_above:
        return "High"
    if value > moderate_above:
        return "Moderate"
    return "Low"


def screen_route(frame: CorridorFrame, route: LineString, layers: dict[str, gpd.GeoDataFrame],
                 shared_corridor_km: float, new_access_km: float,
                 t: EnvThresholds = EnvThresholds(), elevation=None) -> list[dict]:
    """Screen one route against every theme. Returns one dict per theme."""
    elevation = elevation or SyntheticElevation(frame)
    u = {cat: _union(gdf) for cat, gdf in layers.items()}
    L = route.length / 1000.0
    pts, grade, seg_km = _grade_profile(frame, route, elevation)
    results = {}

    def put(key, value, unit, level, explanation):
        results[key] = {"value": value, "unit": unit, "level": level, "explanation": explanation}

    # Wetlands
    wet_in = _km(route.intersection(u["wetland"])) if u["wetland"] else 0.0
    wet_near = _km(route.intersection(u["wetland"].buffer(t.wetland_buffer_m))) - wet_in if u["wetland"] else 0.0
    n_wet = int(layers["wetland"].intersects(route).sum()) if not layers["wetland"].empty else 0
    put("wetlands", wet_in, "km crossed",
        "High" if wet_in > 1 or n_wet >= 2 else ("Moderate" if wet_in > 0 or wet_near > 0.5 else "Low"),
        f"Crosses {n_wet} wetland(s) for {wet_in:.1f} km; a further {wet_near:.1f} km lies within "
        f"{t.wetland_buffer_m:.0f} m of a wetland.")

    # Water
    crossings = 0
    for g in layers["water"].geometry:
        inter = route.intersection(g)
        if not inter.is_empty:
            crossings += len(getattr(inter, "geoms", [inter]))
    riparian = _km(route.intersection(u["water"].buffer(t.riparian_buffer_m))) if u["water"] else 0.0
    put("water", float(crossings), "crossings", _level(crossings, 0, 5),
        f"{crossings} watercourse crossing(s); {riparian:.1f} km within {t.riparian_buffer_m:.0f} m riparian "
        "zones. Watershed boundaries are not assessed.")

    # Protected areas
    prot_in = _km(route.intersection(u["protected"])) if u["protected"] else 0.0
    prot_near = (_km(route.intersection(u["protected"].buffer(t.protected_buffer_m))) - prot_in
                 if u["protected"] else 0.0)
    put("protected", prot_in, "km inside", "High" if prot_in > 0 else ("Moderate" if prot_near > 0 else "Low"),
        f"{prot_in:.1f} km inside and {prot_near:.1f} km within {t.protected_buffer_m / 1000:.0f} km of a "
        "protected area.")

    # Forest
    forest_km = _km(route.intersection(u["forest"])) if u["forest"] else 0.0
    forest_ha = forest_km * t.row_width_m / 10
    put("forest", forest_km, "km cleared", _level(forest_km, 0, 20),
        f"{forest_km:.1f} km through forest: about {forest_ha:,.0f} ha of clearing at a "
        f"{t.row_width_m:.0f} m right-of-way.")

    # Fragmentation: new linear disturbance and intact patches split by the route
    new_linear = max(L - shared_corridor_km, 0.0)
    patches = 0
    for cat in ("forest", "wetland", "habitat"):
        for g in layers[cat].geometry:
            if route.intersects(g):
                pieces = g.difference(route.buffer(1.0))
                if len(getattr(pieces, "geoms", [pieces])) >= 2:
                    patches += 1
    put("fragmentation", new_linear, "km new linear disturbance",
        "High" if patches >= 2 or new_linear > 150 else ("Moderate" if patches >= 1 or new_linear > 50 else "Low"),
        f"{new_linear:.1f} km of new linear disturbance outside existing corridors; splits {patches} "
        "intact forest/wetland/habitat patch(es).")

    # Habitat (only with user data)
    if layer_source(layers, "habitat") == "missing":
        put("habitat", float("nan"), "", "Not assessed",
            "No habitat or biodiversity data has been provided. Effects are unknown, not absent.")
    else:
        hab_in = _km(route.intersection(u["habitat"]))
        hab_near = _km(route.intersection(u["habitat"].buffer(t.habitat_buffer_m))) - hab_in
        put("habitat", hab_in, "km inside", "High" if hab_in > 0 else ("Moderate" if hab_near > 0 else "Low"),
            f"{hab_in:.1f} km inside and {hab_near:.1f} km within {t.habitat_buffer_m:.0f} m of mapped "
            "sensitive habitat (only as complete as the uploaded data).")

    # Terrain proxies
    near_water = (shapely.contains_xy(u["water"].buffer(t.erosion_near_water_m), shapely.get_x(pts),
                                      shapely.get_y(pts)) if u["water"] else np.zeros(len(pts), bool))
    erosion_km = float(((grade > t.erosion_grade_pct) & near_water).sum() * seg_km)
    put("erosion", erosion_km, "km steep near water", _level(erosion_km, 0, 1),
        f"{erosion_km:.1f} km steeper than {t.erosion_grade_pct:.0f}% within {t.erosion_near_water_m:.0f} m of "
        "water. Proxy only: soils and rainfall are not considered.")
    unknown = float(np.isnan(grade[1:]).mean()) if len(grade) > 1 else 0.0
    unknown_note = f" Elevation unknown for {unknown:.0%} of the route." if unknown > 0.001 else ""
    results["erosion"]["explanation"] += unknown_note
    slide_km = float((grade > t.landslide_grade_pct).sum() * seg_km)
    put("landslide", slide_km, "km steep ground", _level(slide_km, 0, 2),
        f"{slide_km:.1f} km steeper than {t.landslide_grade_pct:.0f}%. Proxy only: geology is not assessed."
        + unknown_note)
    flood_km = _km(route.intersection(u["water"].buffer(t.flood_proxy_buffer_m))) if u["water"] else 0.0
    put("flood", flood_km, "km near watercourses", _level(flood_km, 0, 5),
        f"{flood_km:.1f} km within {t.flood_proxy_buffer_m:.0f} m of a watercourse (floodplain proxy). "
        "Mapped floodplains are not used.")

    # Disturbance & restoration
    row_ha = L * t.row_width_m / 10
    access_ha = new_access_km * t.access_road_width_m / 10
    put("disturbance", row_ha + access_ha, "ha", _level(row_ha + access_ha, 100, 500),
        f"About {row_ha:,.0f} ha of right-of-way plus {access_ha:,.0f} ha of new access roads "
        "(temporary workspace not included).")
    restore_ha = (forest_km + wet_in) * t.row_width_m / 10
    put("restoration", restore_ha, "ha", _level(restore_ha, 0, 50),
        f"About {restore_ha:,.0f} ha of forest and wetland will need restoration or reclamation.")

    rows = []
    for theme in THEMES:
        data_status, confidence = confidence_for(theme, layers, elevation.is_synthetic)
        r = results[theme.key]
        if data_status == "Data gap":
            r["level"] = "Not assessed"
        rows.append({"theme": theme.key, "label": theme.label, **r, "data_status": data_status,
                     "confidence": confidence})
    return rows


def screen_routes(frame: CorridorFrame, routes: pd.DataFrame, geoms: dict[str, LineString],
                  layers: dict[str, gpd.GeoDataFrame], t: EnvThresholds = EnvThresholds(),
                  elevation=None) -> pd.DataFrame:
    """Long table: one row per (route, theme)."""
    rows = []
    for _, r in routes.iterrows():
        for res in screen_route(frame, geoms[r["route_id"]], layers, r["shared_corridor_km"], r["new_access_km"], t,
                                elevation):
            rows.append({"route_id": r["route_id"], "route": r["name"], **res})
    return pd.DataFrame(rows)


def level_matrix(screening: pd.DataFrame) -> pd.DataFrame:
    """Themes × routes table of screening levels."""
    m = screening.pivot(index="label", columns="route", values="level")
    return m.reindex([t.label for t in THEMES])


def route_summary(screening: pd.DataFrame) -> pd.DataFrame:
    """Per route: how many themes are High / Moderate / Low / Not assessed, and how many rely on synthetic data."""
    counts = screening.pivot_table(index="route", columns="level", values="theme", aggfunc="count", fill_value=0)
    counts = counts.reindex(columns=LEVELS, fill_value=0)
    counts["Synthetic-only themes"] = (screening.assign(s=screening["confidence"].str.startswith("None"))
                                       .groupby("route")["s"].sum())
    return counts.reset_index()


def conflicts(route: LineString, layers: dict[str, gpd.GeoDataFrame], t: EnvThresholds = EnvThresholds()) -> pd.DataFrame:
    """Specific features a route crosses or passes close to."""
    buffers = {"wetland": t.wetland_buffer_m, "protected": t.protected_buffer_m, "habitat": t.habitat_buffer_m,
               "water": t.riparian_buffer_m, "forest": 0.0}
    rows = []
    for cat, buf in buffers.items():
        gdf = layers.get(cat)
        if gdf is None or gdf.empty:
            continue
        for _, f in gdf.iterrows():
            inside = route.intersection(f.geometry)
            if not inside.is_empty:
                if f.geometry.geom_type in ("LineString", "MultiLineString"):
                    n = len(getattr(inside, "geoms", [inside]))
                    rows.append({"feature": f["name"], "category": cat, "relationship": f"Crossed {n} time(s)",
                                 "km": 0.0, "source": f["source"]})
                else:
                    rows.append({"feature": f["name"], "category": cat, "relationship": "Crossed",
                                 "km": inside.length / 1000, "source": f["source"]})
            elif buf > 0 and route.distance(f.geometry) <= buf:
                rows.append({"feature": f["name"], "category": cat,
                             "relationship": f"Within {buf:,.0f} m ({route.distance(f.geometry):,.0f} m)",
                             "km": 0.0, "source": f["source"]})
    return pd.DataFrame(rows, columns=["feature", "category", "relationship", "km", "source"])
