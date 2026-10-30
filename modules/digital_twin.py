"""3D digital twin: a renderer-independent scene plus a Plotly renderer.

The twin is built in two steps so the 3D engine can be swapped later
(e.g. NVIDIA Omniverse/USD, Blender, CesiumJS) without touching the data:

1. `build_scene(...)`   -> `TwinScene`: plain coordinates (local km east/north of the
                           start point, elevation in metres) for terrain, constraint
                           features, route alternatives, pipeline sections, sensors and
                           indicative facilities. `scene_to_json` exports it.
2. `render_plotly(...)` -> Plotly figure for one of three modes:
     * Planning      - terrain, features, all route alternatives, indicative facilities
     * Construction  - the modelled route coloured by construction stage per section
     * Operations    - the modelled route coloured by monitoring status, sensor readings

Coordinates come from the project's real start/destination (UTM). Terrain comes from
the project's elevation source: an uploaded DEM, or clearly labelled synthetic terrain.
"""

import json
from dataclasses import dataclass, field

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shapely
from pyproj import Transformer
from shapely.geometry import LineString

from modules.config import STATUS_COLORS
from modules.gis_layers import CATEGORIES, CorridorFrame
from modules.maps import LAYER_STYLES

BURIAL_DEPTH_M = 1.5  # typical depth of cover (demonstration value)
FEATURE_LIFT_M = 4.0  # draw features slightly above the ground so they stay visible
MODES = ["Planning", "Construction", "Operations"]

CONSTRUCTION_STAGES = ["Not started", "Right-of-way clearing", "Pipe strung & welded", "Lowered & backfilled",
                       "Hydrotested", "Complete"]
# Ordered progress -> one hue, light to dark ("Not started" is neutral grey).
STAGE_COLORS = ["#C4C7CC", "#BFD4EE", "#8DB4E3", "#5B92D2", "#2F6DB8", "#1B4886"]

FACILITY_NAMES = {"Oil": "Pump station", "Natural gas": "Compressor station", "Water": "Pump station"}
TERRAIN_COLORSCALE = [[0.0, "#ECE7DB"], [1.0, "#A89A7C"]]


@dataclass
class TwinScene:
    crs_epsg: int
    origin_x: float  # UTM metres of the start point (local 0, 0)
    origin_y: float
    terrain: dict | None  # {"x_km": 1-D, "y_km": 1-D, "z_m": 2-D}
    terrain_label: str
    terrain_synthetic: bool
    features: list[dict] = field(default_factory=list)  # category, name, synthetic, x_km, y_km, z_m
    routes: list[dict] = field(default_factory=list)  # route_id, name, color, x_km, y_km, z_m, chainage_km
    pipeline_route_id: str = ""
    sections: pd.DataFrame | None = None  # section geometry + sensor positions
    facilities: list[dict] = field(default_factory=list)


def _drape(elevation, x, y, lift=0.0):
    return elevation.sample(np.asarray(x), np.asarray(y)) + lift


def _line_xyz(line: LineString, elevation, origin, step_m=250.0, lift=0.0):
    dense = shapely.segmentize(line, step_m)
    xy = np.asarray(dense.coords)
    z = _drape(elevation, xy[:, 0], xy[:, 1], lift)
    return (xy[:, 0] - origin[0]) / 1000, (xy[:, 1] - origin[1]) / 1000, z


def _outlines(geom):
    """Lines to draw for a feature: the line itself, or polygon boundaries."""
    if geom.geom_type in ("LineString", "MultiLineString"):
        return list(getattr(geom, "geoms", [geom]))
    boundary = geom.boundary
    return list(getattr(boundary, "geoms", [boundary]))


def build_scene(frame: CorridorFrame, routes: pd.DataFrame, geoms: dict[str, LineString],
                layers: dict[str, gpd.GeoDataFrame], elevation, pipeline_route_id: str,
                sections: pd.DataFrame, route_colors: dict[str, str], pipeline_type: str,
                terrain_cells: int = 70) -> TwinScene:
    origin = (frame.x0, frame.y0)
    to_ll = Transformer.from_crs(frame.crs_epsg, 4326, always_xy=True)

    minx, miny, maxx, maxy = frame.study_area().bounds
    xs = np.linspace(minx, maxx, terrain_cells)
    ys = np.linspace(miny, maxy, terrain_cells)
    gx, gy = np.meshgrid(xs, ys)
    terrain = {"x_km": (xs - origin[0]) / 1000, "y_km": (ys - origin[1]) / 1000, "z_m": elevation.sample(gx, gy)}

    scene = TwinScene(frame.crs_epsg, origin[0], origin[1], terrain, elevation.label, elevation.is_synthetic,
                      pipeline_route_id=pipeline_route_id)

    for cat, gdf in layers.items():
        for _, f in gdf.iterrows():
            for part in _outlines(f.geometry):
                x, y, z = _line_xyz(part, elevation, origin, lift=FEATURE_LIFT_M)
                scene.features.append({"category": cat, "name": f["name"], "synthetic": bool(f["synthetic"]),
                                       "x_km": x, "y_km": y, "z_m": z})

    for _, r in routes.iterrows():
        line = geoms[r["route_id"]]
        x, y, z = _line_xyz(line, elevation, origin, step_m=200.0, lift=-BURIAL_DEPTH_M)
        chainage = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
        scene.routes.append({"route_id": r["route_id"], "name": r["name"], "color": route_colors[r["route_id"]],
                             "x_km": x, "y_km": y, "z_m": z, "chainage_km": chainage})

    # Sections and sensors are placed by chainage along the modelled pipeline route.
    line = geoms[pipeline_route_id]
    scale = line.length / 1000 / max(float(sections["end_km"].iloc[-1]), 1e-9)  # guards rounding differences
    pts = shapely.line_interpolate_point(line, sections["sensor_km"].to_numpy() * scale * 1000)
    sx, sy = shapely.get_x(pts), shapely.get_y(pts)
    lon, lat = to_ll.transform(sx, sy)
    sec = sections.copy()
    sec["x_km"], sec["y_km"] = (sx - origin[0]) / 1000, (sy - origin[1]) / 1000
    sec["z_m"] = _drape(elevation, sx, sy, -BURIAL_DEPTH_M)
    sec["lat"], sec["lon"] = lat, lon
    scene.sections = sec

    # Indicative facilities: a station at the start, a terminal at the end, block valves between sections.
    station = FACILITY_NAMES.get(pipeline_type, "Pump station")
    for kind, name, km in ([("station", f"{station} (indicative)", 0.0)]
                           + [("valve", f"Block valve BV-{i} (indicative)", float(km))
                              for i, km in enumerate(sections["start_km"].iloc[1:], start=1)]
                           + [("terminal", "Delivery terminal (indicative)", float(sections["end_km"].iloc[-1]))]):
        p = line.interpolate(min(km * scale * 1000, line.length))
        scene.facilities.append({"kind": kind, "name": name, "chainage_km": km,
                                 "x_km": (p.x - origin[0]) / 1000, "y_km": (p.y - origin[1]) / 1000,
                                 "z_m": float(_drape(elevation, [p.x], [p.y])[0])})
    return scene


def construction_progress_pct(sections: pd.DataFrame, progress: dict[str, str]) -> float:
    """Length-weighted share of the work done (each stage counts equally)."""
    lengths = (sections["end_km"] - sections["start_km"]).to_numpy()
    done = np.array([CONSTRUCTION_STAGES.index(progress.get(s, CONSTRUCTION_STAGES[0]))
                     for s in sections["section_id"]]) / (len(CONSTRUCTION_STAGES) - 1)
    return float((lengths * done).sum() / lengths.sum() * 100) if lengths.sum() else 0.0


def _section_xyz(scene: TwinScene, start_km: float, end_km: float):
    route = next(r for r in scene.routes if r["route_id"] == scene.pipeline_route_id)
    total = route["chainage_km"][-1]
    sec_total = float(scene.sections["end_km"].iloc[-1])
    a, b = start_km / sec_total * total, end_km / sec_total * total
    keep = (route["chainage_km"] >= a) & (route["chainage_km"] <= b)
    # Include interpolated end points so neighbouring sections meet.
    xs = np.concatenate([[np.interp(a, route["chainage_km"], route["x_km"])], route["x_km"][keep],
                         [np.interp(b, route["chainage_km"], route["x_km"])]])
    ys = np.concatenate([[np.interp(a, route["chainage_km"], route["y_km"])], route["y_km"][keep],
                         [np.interp(b, route["chainage_km"], route["y_km"])]])
    zs = np.concatenate([[np.interp(a, route["chainage_km"], route["z_m"])], route["z_m"][keep],
                         [np.interp(b, route["chainage_km"], route["z_m"])]])
    return xs, ys, zs


def render_plotly(scene: TwinScene, mode: str = "Planning", statuses: pd.DataFrame | None = None,
                  snapshot: pd.DataFrame | None = None, progress: dict | None = None,
                  selected_section: str | None = None, show_terrain: bool = True, show_features: bool = True,
                  show_alternatives: bool = True, vertical_exaggeration: float = 15.0,
                  flow_unit: str = "") -> go.Figure:
    fig = go.Figure()
    progress = progress or {}
    status_by = statuses.set_index("section_id")["status"].to_dict() if statuses is not None else {}
    snap = snapshot.set_index("section_id") if snapshot is not None and not snapshot.empty else None

    if show_terrain and scene.terrain is not None:
        fig.add_trace(go.Surface(
            x=scene.terrain["x_km"], y=scene.terrain["y_km"], z=scene.terrain["z_m"],
            colorscale=TERRAIN_COLORSCALE, opacity=0.9, showscale=False, hoverinfo="skip",
            name="Terrain", lighting={"ambient": 0.75, "diffuse": 0.6}, contours={"z": {"show": False}},
        ))

    if show_features:
        by_cat: dict[str, list[dict]] = {}
        for f in scene.features:
            by_cat.setdefault(f["category"], []).append(f)
        for cat, items in by_cat.items():
            xs, ys, zs, text = [], [], [], []
            for f in items:
                xs += [*f["x_km"], None]
                ys += [*f["y_km"], None]
                zs += [*f["z_m"], None]
                text += [f["name"]] * len(f["x_km"]) + [None]
            synthetic = any(f["synthetic"] for f in items)
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="lines", text=text, hovertemplate="%{text}<extra></extra>",
                line={"color": LAYER_STYLES[cat]["color"], "width": 3, "dash": "dash" if synthetic else "solid"},
                name=CATEGORIES[cat]["label"] + (" (SYNTHETIC)" if synthetic else ""),
            ))

    pipeline = next(r for r in scene.routes if r["route_id"] == scene.pipeline_route_id)
    if mode == "Planning":
        for r in scene.routes:
            is_pipe = r["route_id"] == scene.pipeline_route_id
            if not is_pipe and not show_alternatives:
                continue
            fig.add_trace(go.Scatter3d(
                x=r["x_km"], y=r["y_km"], z=r["z_m"], mode="lines", name=r["name"] + (" (modelled)" if is_pipe else ""),
                line={"color": r["color"], "width": 9 if is_pipe else 4, "dash": "dash" if r["route_id"] == "REF" else "solid"},
                customdata=r["chainage_km"], hovertemplate=f"{r['name']}<br>KP %{{customdata:.1f}} km<br>"
                                                           "Pipe elevation %{z:.0f} m<extra></extra>",
            ))
    else:
        legend_done = set()
        for _, sec in scene.sections.iterrows():
            sid = sec["section_id"]
            if mode == "Construction":
                stage = progress.get(sid, CONSTRUCTION_STAGES[0])
                label, color = stage, STAGE_COLORS[CONSTRUCTION_STAGES.index(stage)]
            else:
                label = status_by.get(sid, "Normal")
                color = STATUS_COLORS[label]
            x, y, z = _section_xyz(scene, sec["start_km"], sec["end_km"])
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode="lines", name=label, legendgroup=label, showlegend=label not in legend_done,
                line={"color": color, "width": 14 if sid == selected_section else 8},
                hovertemplate=f"<b>{sid}</b> · {label}<br>KP {sec['start_km']:.1f}–{sec['end_km']:.1f} km<extra></extra>",
            ))
            legend_done.add(label)

    # Facilities (planning & construction) or sensors (operations)
    lift = 20.0 * max(1.0, 15.0 / max(vertical_exaggeration, 0.1))
    if mode in ("Planning", "Construction"):
        fac = scene.facilities
        fig.add_trace(go.Scatter3d(
            x=[f["x_km"] for f in fac], y=[f["y_km"] for f in fac], z=[f["z_m"] + lift for f in fac],
            mode="markers", name="Facilities (indicative)",
            marker={"size": [9 if f["kind"] != "valve" else 5 for f in fac], "symbol": "square",
                    "color": "#111827", "line": {"color": "white", "width": 1}},
            text=[f"{f['name']}<br>KP {f['chainage_km']:.1f} km" for f in fac], hoverinfo="text",
        ))
    if mode in ("Construction", "Operations"):
        s = scene.sections
        hover, colors = [], []
        for _, sec in s.iterrows():
            sid = sec["section_id"]
            status = status_by.get(sid, "Normal")
            text = (f"<b>Sensor station {sid}</b><br>KP {sec['sensor_km']:.1f} km"
                    f"<br>{sec['lat']:.5f}, {sec['lon']:.5f}")
            if mode == "Operations":
                text += f"<br>Status: {status}"
                if snap is not None and sid in snap.index:
                    r = snap.loc[sid]
                    text += (f"<br>Pressure: {r['pressure_kpa']:,.0f} kPa<br>Temperature: {r['temperature_c']:.1f} °C"
                             f"<br>Flow in/out: {r['flow_in']:,.0f} / {r['flow_out']:,.0f} {flow_unit}"
                             f"<br>Sensor health: {r['sensor_health']:.0f}/100")
                colors.append(STATUS_COLORS[status])
            else:
                colors.append("#111827")
            hover.append(text)
        fig.add_trace(go.Scatter3d(
            x=s["x_km"], y=s["y_km"], z=s["z_m"] + lift, mode="markers+text", name="Sensor stations",
            text=s["section_id"], textposition="top center", textfont={"size": 10, "color": "#111827"},
            marker={"size": [10 if sid == selected_section else 6 for sid in s["section_id"]], "symbol": "diamond",
                    "color": colors, "line": {"color": "#1F2937", "width": 1}},
            hovertext=hover, hoverinfo="text",
        ))

    # Keep real horizontal proportions; exaggerate the vertical so relief is visible.
    x_all = np.concatenate([pipeline["x_km"], scene.terrain["x_km"] if scene.terrain else []])
    y_all = np.concatenate([pipeline["y_km"], scene.terrain["y_km"] if scene.terrain else []])
    z_vals = scene.terrain["z_m"] if scene.terrain is not None else pipeline["z_m"]
    span = max(np.ptp(x_all), np.ptp(y_all), 1e-6)
    z_span_km = max(float(np.nanmax(z_vals) - np.nanmin(z_vals)) / 1000 if np.isfinite(z_vals).any() else 0.1, 0.01)
    fig.update_layout(
        height=640, margin={"l": 0, "r": 0, "t": 10, "b": 0},
        legend={"orientation": "h", "y": 1.02, "x": 0},
        scene={
            "xaxis": {"title": "East of start (km)"}, "yaxis": {"title": "North of start (km)"},
            "zaxis": {"title": "Elevation (m)"},
            "aspectmode": "manual",
            "aspectratio": {"x": 2.2 * np.ptp(x_all) / span, "y": 2.2 * np.ptp(y_all) / span,
                            "z": max(0.05, 2.2 * z_span_km * vertical_exaggeration / span)},
            "camera": {"eye": {"x": -0.5, "y": -1.35, "z": 0.75}, "center": {"x": 0, "y": 0, "z": -0.1}},
        },
    )
    return fig


def scene_to_json(scene: TwinScene) -> str:
    """Export the scene for other 3D tools (local km east/north of origin; elevation in metres)."""
    def arr(a):
        return [None if not np.isfinite(v) else round(float(v), 4) for v in np.asarray(a, float).ravel()]

    out = {
        "format": "flowguard-twin-scene/1",
        "crs": f"EPSG:{scene.crs_epsg}", "origin_utm_m": [scene.origin_x, scene.origin_y],
        "units": {"x_y": "km from origin", "z": "m"},
        "terrain": {"label": scene.terrain_label, "synthetic": scene.terrain_synthetic,
                    "x_km": arr(scene.terrain["x_km"]), "y_km": arr(scene.terrain["y_km"]),
                    "z_m_rows": [arr(row) for row in scene.terrain["z_m"]]} if scene.terrain else None,
        "features": [{"category": f["category"], "name": f["name"], "synthetic": f["synthetic"],
                      "x_km": arr(f["x_km"]), "y_km": arr(f["y_km"]), "z_m": arr(f["z_m"])} for f in scene.features],
        "routes": [{"route_id": r["route_id"], "name": r["name"], "x_km": arr(r["x_km"]), "y_km": arr(r["y_km"]),
                    "z_m": arr(r["z_m"])} for r in scene.routes],
        "pipeline_route_id": scene.pipeline_route_id,
        "sections": json.loads(scene.sections.to_json(orient="records")) if scene.sections is not None else [],
        "facilities": [{**f, "indicative": True} for f in scene.facilities],
    }
    return json.dumps(out)
