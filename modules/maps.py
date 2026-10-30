"""Folium (Leaflet) map builders for project setup and route planning.

Maps are drawn in latitude/longitude. Synthetic features are drawn with dashed
outlines and "SYNTHETIC" in their names and tooltips, so they cannot be
mistaken for real geography.

Basemap tiles come from OpenStreetMap (free, needs an internet connection;
please respect the OSM tile usage policy). Choose "None" to work offline.
"""

import folium
import geopandas as gpd
import pandas as pd

from modules.gis_layers import CATEGORIES
from modules.projects import Project

# Fixed categorical order (validated palette); assigned by route position, never cycled.
ROUTE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
REFERENCE_COLOR = "#52514e"

LAYER_STYLES = {
    "water": {"color": "#1f6fb2", "fillColor": "#5aa9e6", "weight": 2, "fillOpacity": 0.35},
    "wetland": {"color": "#2a9d8f", "fillColor": "#7fd1c7", "weight": 1, "fillOpacity": 0.45},
    "forest": {"color": "#2f6b3a", "fillColor": "#6fa86f", "weight": 1, "fillOpacity": 0.30},
    "protected": {"color": "#b42318", "fillColor": "#f04438", "weight": 2, "fillOpacity": 0.18},
    "road": {"color": "#4b5563", "weight": 3, "fillOpacity": 0},
    "habitat": {"color": "#7a3e9d", "fillColor": "#b98bd6", "weight": 2, "fillOpacity": 0.25},
}

BASEMAPS = {"OpenStreetMap (online)": "OpenStreetMap", "None (offline)": None}


def route_colors(route_ids: list[str]) -> dict[str, str]:
    colors, i = {}, 0
    for rid in route_ids:
        if rid == "REF":
            colors[rid] = REFERENCE_COLOR
        else:
            colors[rid] = ROUTE_COLORS[i % len(ROUTE_COLORS)]
            i += 1
    return colors


def _base_map(project: Project, basemap: str) -> folium.Map:
    center = [(project.start_lat + project.end_lat) / 2, (project.start_lon + project.end_lon) / 2]
    # scrollWheelZoom off: scrolling the page over the map should scroll the page (zoom with +/-).
    m = folium.Map(location=center, zoom_start=8, tiles=BASEMAPS.get(basemap, "OpenStreetMap"),
                   control_scale=True, scrollWheelZoom=False)
    m.fit_bounds([[project.start_lat, project.start_lon], [project.end_lat, project.end_lon]], padding=(40, 40))
    return m


def _endpoints(m: folium.Map, project: Project) -> None:
    folium.Marker([project.start_lat, project.start_lon], tooltip=f"Start: {project.start_name}",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([project.end_lat, project.end_lon], tooltip=f"Destination: {project.end_name}",
                  icon=folium.Icon(color="red", icon="flag")).add_to(m)


def project_map(project: Project, basemap: str = "OpenStreetMap (online)") -> folium.Map:
    """Start/destination markers joined by a dashed straight line."""
    m = _base_map(project, basemap)
    folium.PolyLine([[project.start_lat, project.start_lon], [project.end_lat, project.end_lon]],
                    color=REFERENCE_COLOR, weight=2, dash_array="6 6",
                    tooltip=f"Straight line: {project.straight_line_km:,.1f} km").add_to(m)
    _endpoints(m, project)
    return m


def route_map(project: Project, layers_wgs84: dict[str, gpd.GeoDataFrame], routes: pd.DataFrame,
              route_lines: dict[str, list[tuple[float, float]]], basemap: str = "OpenStreetMap (online)",
              highlight: str | None = None) -> folium.Map:
    """Constraint layers + route alternatives, each toggleable in the layer control."""
    m = _base_map(project, basemap)
    for cat, gdf in layers_wgs84.items():
        if gdf.empty:
            continue
        style = LAYER_STYLES[cat]
        synthetic = bool(gdf["synthetic"].any())
        label = CATEGORIES[cat]["label"] + (" (SYNTHETIC)" if synthetic else "")
        folium.GeoJson(
            gdf[["name", "source", "geometry"]],
            name=label,
            style_function=lambda _f, s=style, dash=synthetic: {**s, "dashArray": "6 4" if dash else None},
            tooltip=folium.GeoJsonTooltip(fields=["name", "source"], aliases=["Feature", "Source"]),
        ).add_to(m)

    colors = route_colors(routes["route_id"].tolist())
    for _, r in routes.iterrows():
        rid = r["route_id"]
        is_ref = rid == "REF"
        group = folium.FeatureGroup(name=f"Route {r['name']}")
        folium.PolyLine(
            route_lines[rid], color=colors[rid], weight=7 if rid == highlight else (2 if is_ref else 4),
            opacity=0.95, dash_array="8 6" if is_ref else None,
            tooltip=f"{r['name']} · {r['length_km']:,.1f} km",
        ).add_to(group)
        group.add_to(m)

    _endpoints(m, project)
    folium.LayerControl(collapsed=True).add_to(m)
    return m
