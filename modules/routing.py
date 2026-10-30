"""Corridor generation (least-cost paths) and route measurement.

How alternatives are generated
------------------------------
1. The study area is divided into a grid of square cells (in metres).
2. Each cell gets a "cost of passing through", built from the constraint layers
   and terrain slope. Different *strategies* weight these differently
   (e.g. one strongly avoids wetlands, another follows existing corridors).
3. Dijkstra's shortest-path algorithm finds the cheapest path from start to
   destination for each strategy.
4. Protected areas are hard constraints: their cells are impassable when
   "avoid hard constraints" is on.

The straight line is added as a reference only: it shows what is avoided
by the other alternatives. Every route is then measured against the layers
with real geometry operations (lengths of overlap, crossing counts, slopes).

This is a screening-level tool. Routes are indicative corridors, not designs.
"""

import heapq
import math
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from modules.config import SOCIAL_NOT_ASSESSED
from modules.elevation import SyntheticElevation
from modules.gis_layers import CorridorFrame

# Transparent heuristics used when measuring routes (shown in the app).
MEASUREMENT_ASSUMPTIONS = {
    "shared_corridor_buffer_m": 300,  # within this distance of an existing road counts as "shared corridor"
    "access_free_distance_m": 2000,  # route parts farther than this from a road need new access
    "access_road_ratio": 0.25,  # km of new access road per km of route needing access
    "steep_grade_pct": 15.0,  # grades above this are flagged as construction challenges
    "profile_sample_m": 100,  # elevation sampled every 100 m along a route
}


@dataclass(frozen=True)
class RoutingStrategy:
    key: str
    name: str
    description: str
    slope_weight: float  # extra cost per 10% slope
    water_weight: float
    wetland_weight: float
    forest_weight: float
    corridor_bonus: float  # 0-0.9: cost reduction near existing roads/corridors
    habitat_weight: float = 0.0  # only has an effect when habitat data has been uploaded


STRATEGIES: list[RoutingStrategy] = [
    RoutingStrategy("direct", "Most direct feasible", "Shortest path that still respects hard constraints.",
                    0.0, 0.0, 0.0, 0.0, 0.0),
    RoutingStrategy("environment", "Lower environmental disturbance",
                    "Strongly avoids wetlands, water crossings, forest clearing and (if supplied) sensitive habitat.",
                    0.3, 6.0, 10.0, 2.0, 0.0, habitat_weight=10.0),
    RoutingStrategy("terrain", "Gentler terrain", "Avoids steep slopes to reduce construction difficulty.",
                    6.0, 1.0, 2.0, 0.0, 0.0),
    RoutingStrategy("corridor", "Follow existing corridor",
                    "Parallels existing roads/utility corridors to share access and limit new disturbance.",
                    0.3, 1.0, 3.0, 0.5, 0.8, habitat_weight=3.0),
]
STRATEGIES_BY_KEY = {s.key: s for s in STRATEGIES}


# ---------------------------------------------------------------------------
# Cost grid
# ---------------------------------------------------------------------------
@dataclass
class CostGrid:
    xs: np.ndarray  # cell-centre x coordinates (columns)
    ys: np.ndarray  # cell-centre y coordinates (rows)
    cell_m: float
    inside: np.ndarray  # bool: cell lies inside the study area
    masks: dict[str, np.ndarray]  # bool masks per category (+ "near_road")
    slope_pct: np.ndarray


def _mask(geoms: gpd.GeoSeries, gx: np.ndarray, gy: np.ndarray, line_halfwidth: float) -> np.ndarray:
    """Which grid cells touch the features? Lines are widened slightly so crossings register."""
    if geoms.empty:
        return np.zeros(gx.shape, dtype=bool)
    shapes = [g.buffer(line_halfwidth) if g.geom_type in ("LineString", "MultiLineString") else g for g in geoms]
    merged = unary_union(shapes)
    return shapely.contains_xy(merged, gx, gy)


def build_cost_grid(frame: CorridorFrame, layers: dict[str, gpd.GeoDataFrame], elevation=None,
                    target_cells: int = 240) -> CostGrid:
    """Rasterise the layers onto a grid whose longer side has about `target_cells` cells."""
    area = frame.study_area()
    minx, miny, maxx, maxy = area.bounds
    cell = max(maxx - minx, maxy - miny) / target_cells
    xs = np.arange(minx + cell / 2, maxx, cell)
    ys = np.arange(miny + cell / 2, maxy, cell)
    gx, gy = np.meshgrid(xs, ys)
    masks = {cat: _mask(gdf.geometry, gx, gy, cell * 0.75) for cat, gdf in layers.items()}
    road_buffer = max(cell * 1.5, MEASUREMENT_ASSUMPTIONS["shared_corridor_buffer_m"])
    masks["near_road"] = _mask(layers["road"].geometry, gx, gy, road_buffer)
    # Hard constraints get a one-cell safety margin so smoothed lines cannot clip their edges.
    masks["protected"] = _mask(layers["protected"].geometry.buffer(cell), gx, gy, cell)
    elev = (elevation or SyntheticElevation(frame)).sample(gx, gy)
    dzdy, dzdx = np.gradient(elev, cell)
    # Where elevation is unknown (outside DEM coverage) no slope penalty is applied.
    slope = np.nan_to_num(100 * np.hypot(dzdx, dzdy), nan=0.0)
    return CostGrid(xs, ys, cell, shapely.contains_xy(area, gx, gy), masks, slope)


def cost_surface(grid: CostGrid, strategy: RoutingStrategy, avoid_hard_constraints: bool = True) -> np.ndarray:
    m = grid.masks
    cost = (1.0
            + strategy.slope_weight * grid.slope_pct / 10.0
            + strategy.water_weight * m["water"]
            + strategy.wetland_weight * m["wetland"]
            + strategy.forest_weight * m["forest"]
            + strategy.habitat_weight * m["habitat"])
    cost = cost * np.where(m["near_road"], 1.0 - strategy.corridor_bonus, 1.0)
    cost = np.where(grid.inside, cost, np.inf)
    if avoid_hard_constraints:
        cost = np.where(m["protected"], np.inf, cost)
    return cost


# ---------------------------------------------------------------------------
# Least-cost path (Dijkstra on an 8-connected grid)
# ---------------------------------------------------------------------------
_MOVES = [(-1, -1, math.sqrt(2)), (-1, 0, 1.0), (-1, 1, math.sqrt(2)), (0, -1, 1.0),
          (0, 1, 1.0), (1, -1, math.sqrt(2)), (1, 0, 1.0), (1, 1, math.sqrt(2))]


def least_cost_path(cost: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]] | None:
    """Cheapest chain of cells from start to end, or None if the end cannot be reached.

    Moving between two cells costs the average of their costs times the step length
    (1 for side neighbours, √2 for diagonal neighbours).
    """
    nrows, ncols = cost.shape
    flat = cost.ravel().tolist()
    start_i, end_i = start[0] * ncols + start[1], end[0] * ncols + end[1]
    dist = [math.inf] * len(flat)
    prev = [-1] * len(flat)
    dist[start_i] = 0.0
    heap = [(0.0, start_i)]
    while heap:
        d, i = heapq.heappop(heap)
        if i == end_i:
            break
        if d > dist[i]:
            continue
        r, c = divmod(i, ncols)
        ci = flat[i]
        for dr, dc, step in _MOVES:
            rr, cc = r + dr, c + dc
            if 0 <= rr < nrows and 0 <= cc < ncols:
                j = rr * ncols + cc
                cj = flat[j]
                if cj == math.inf:
                    continue
                nd = d + 0.5 * (ci + cj) * step
                if nd < dist[j]:
                    dist[j] = nd
                    prev[j] = i
                    heapq.heappush(heap, (nd, j))
    if dist[end_i] == math.inf:
        return None
    path, i = [], end_i
    while i != -1:
        path.append(divmod(i, ncols))
        i = prev[i]
    return path[::-1]


def _nearest_cell(grid: CostGrid, x: float, y: float) -> tuple[int, int]:
    return int(np.abs(grid.ys - y).argmin()), int(np.abs(grid.xs - x).argmin())


def generate_route(frame: CorridorFrame, grid: CostGrid, strategy: RoutingStrategy,
                   avoid_hard_constraints: bool = True) -> LineString | None:
    """Least-cost corridor for one strategy, as a simplified line in projected metres."""
    cost = cost_surface(grid, strategy, avoid_hard_constraints)
    start, end = _nearest_cell(grid, frame.x0, frame.y0), _nearest_cell(grid, frame.x1, frame.y1)
    for r, c in (start, end):  # end points are always usable, even inside a constraint
        if not np.isfinite(cost[r, c]):
            cost[r, c] = 1.0
    cells = least_cost_path(cost, start, end)
    if cells is None:
        return None
    coords = [(grid.xs[c], grid.ys[r]) for r, c in cells]
    coords[0], coords[-1] = (frame.x0, frame.y0), (frame.x1, frame.y1)
    # Simplifying removes most of the grid "staircase" without cutting corners across constraints.
    return LineString(coords).simplify(grid.cell_m * 0.6, preserve_topology=True)


# ---------------------------------------------------------------------------
# Route measurement
# ---------------------------------------------------------------------------
def _union(gdf: gpd.GeoDataFrame):
    return unary_union(list(gdf.geometry)) if not gdf.empty else None


def _overlap_km(route: LineString, geom) -> float:
    return 0.0 if geom is None else route.intersection(geom).length / 1000.0


def _count_crossings(route: LineString, water: gpd.GeoDataFrame) -> int:
    """Each place where the route crosses a river line or enters a lake/river polygon counts once."""
    count = 0
    for geom in water.geometry:
        inter = route.intersection(geom)
        if inter.is_empty:
            continue
        if geom.geom_type in ("LineString", "MultiLineString"):
            count += len(getattr(inter, "geoms", [inter]))
        else:
            count += len([g for g in getattr(inter, "geoms", [inter]) if g.length > 0])
    return count


def elevation_profile(frame: CorridorFrame, route: LineString, elevation=None) -> pd.DataFrame:
    """Elevation and grade every 100 m along a route (NaN where elevation is unknown)."""
    step = MEASUREMENT_ASSUMPTIONS["profile_sample_m"]
    distances = np.append(np.arange(0, route.length, step), route.length)
    pts = shapely.line_interpolate_point(route, distances)
    x, y = shapely.get_x(pts), shapely.get_y(pts)
    elev = (elevation or SyntheticElevation(frame)).sample(x, y)
    grade = np.zeros_like(elev)
    grade[1:] = 100 * np.diff(elev) / np.maximum(np.diff(distances), 1e-6)
    return pd.DataFrame({"chainage_km": distances / 1000, "elevation_m": elev, "grade_pct": grade})


def measure_route(frame: CorridorFrame, route: LineString, layers: dict[str, gpd.GeoDataFrame],
                  elevation=None) -> dict:
    """Measure one route against every layer. All values come from geometry, not guesses."""
    a = MEASUREMENT_ASSUMPTIONS
    road = _union(layers["road"])
    profile = elevation_profile(frame, route, elevation)
    abs_grade = profile["grade_pct"].abs()
    known = abs_grade.notna().mean()  # share of the route with known elevation
    length_km = route.length / 1000.0
    wetland_km = _overlap_km(route, _union(layers["wetland"]))
    protected_km = _overlap_km(route, _union(layers["protected"]))
    shared = _overlap_km(route, road.buffer(a["shared_corridor_buffer_m"])) if road is not None else 0.0
    needs_access = (route.difference(road.buffer(a["access_free_distance_m"])).length / 1000.0
                    if road is not None else length_km)
    steep_km = float((abs_grade > a["steep_grade_pct"]).sum() * a["profile_sample_m"] / 1000.0)
    # Terrain difficulty 1-5: driven by average grade, plus soft ground (wetland share).
    mean_grade = abs_grade.mean() if known > 0 else 0.0
    difficulty = float(np.clip(1 + mean_grade * 0.6 + 6 * wetland_km / max(length_km, 1e-6), 1, 5))

    hazards = []
    if steep_km > 0:
        hazards.append(f"{steep_km:.1f} km steeper than {a['steep_grade_pct']:.0f}% grade")
    if wetland_km > 0:
        hazards.append(f"{wetland_km:.1f} km of soft/wet ground")
    if known < 0.999:
        hazards.append(f"elevation unknown for {1 - known:.0%} of the route (outside DEM coverage)")
    hazards.append("geotechnical, landslide and flood exposure not assessed")

    protected_names = []
    if not layers["protected"].empty:
        hit = layers["protected"][layers["protected"].intersects(route)]
        protected_names = hit["name"].tolist()

    return {
        "length_km": length_km,
        "terrain_difficulty": round(difficulty, 2),
        "max_slope_pct": float(abs_grade.max()) if known > 0 else float("nan"),
        "water_crossings": _count_crossings(route, layers["water"]),
        "wetland_km": wetland_km,
        "forest_clearing_km": _overlap_km(route, _union(layers["forest"])),
        "new_access_km": needs_access * a["access_road_ratio"],
        "shared_corridor_km": shared,
        "protected_km": protected_km,
        "elevation_gain_m": float(np.nansum(np.clip(np.diff(profile["elevation_m"]), 0, None))),
        "hard_constraints": (f"Crosses {', '.join(protected_names)} ({protected_km:.1f} km)"
                             if protected_km > 0 else ""),
        "geohazards": "; ".join(hazards),
    }


def generate_alternatives(frame: CorridorFrame, layers: dict[str, gpd.GeoDataFrame],
                          strategy_keys: list[str], avoid_hard_constraints: bool = True,
                          data_quality: str = "Synthetic demonstration",
                          elevation=None) -> tuple[pd.DataFrame, dict, list[str]]:
    """Generate, de-duplicate and measure route alternatives.

    Returns (routes table, {route_id: LineString in projected CRS}, messages).
    """
    grid = build_cost_grid(frame, layers, elevation)
    candidates = [("REF", "Straight-line reference",
                   "Direct line for comparison only: ignores all constraints.",
                   LineString([(frame.x0, frame.y0), (frame.x1, frame.y1)]))]
    messages = []
    for key in strategy_keys:
        s = STRATEGIES_BY_KEY[key]
        line = generate_route(frame, grid, s, avoid_hard_constraints)
        if line is None:
            messages.append(f"No feasible path for '{s.name}': constraints block every route.")
            continue
        candidates.append((key, s.name, s.description, line))

    rows, geoms = [], {}
    for i, (key, name, desc, line) in enumerate(candidates):
        # Skip strategies that produced (almost) the same corridor as an earlier one.
        dup = next((g for g in geoms.values() if line.hausdorff_distance(g) < grid.cell_m * 3), None)
        if dup is not None and key != "REF":
            messages.append(f"'{name}' produced nearly the same corridor as another alternative and was merged.")
            continue
        route_id = "REF" if key == "REF" else f"R{len(geoms)}"
        geoms[route_id] = line
        rows.append({"route_id": route_id, "strategy": key, "name": f"{route_id}: {name}", "description": desc,
                     **measure_route(frame, line, layers, elevation)})
    routes = pd.DataFrame(rows)
    routes["social_indigenous"] = SOCIAL_NOT_ASSESSED
    routes["cultural_heritage"] = "Not assessed: authorised information required"
    routes["data_quality"] = data_quality
    return routes, geoms, messages


def routes_to_wgs84(geoms: dict[str, LineString], crs_epsg: int) -> dict[str, list[tuple[float, float]]]:
    """Convert projected route lines to [(lat, lon), ...] lists for map display."""
    to_ll = Transformer.from_crs(crs_epsg, 4326, always_xy=True)
    out = {}
    for rid, line in geoms.items():
        lon, lat = to_ll.transform(*np.asarray(line.coords).T)
        out[rid] = list(zip(lat, lon))
    return out


def point_wgs84(frame: CorridorFrame, x: float, y: float) -> Point:
    lon, lat = Transformer.from_crs(frame.crs_epsg, 4326, always_xy=True).transform(x, y)
    return Point(lon, lat)
