"""Route Planning (Module B): generated corridors on a map, measured and compared."""

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from modules import projects as store
from modules import ui
from modules.charts import elevation_profile_chart, route_score_chart, route_tradeoff_heatmap
from modules.config import ESTIMATE_DISCLAIMER
from modules.elevation import delete_dem, get_dem, save_dem
from modules.gis_layers import CATEGORIES, corridor_frame, prepare_user_layer, read_uploaded_layer
from modules.maps import BASEMAPS, route_colors, route_map
from modules.route_engine import CRITERIA, explain_tradeoffs, score_routes
from modules.routing import MEASUREMENT_ASSUMPTIONS, STRATEGIES

ss = st.session_state
project = ui.get_active_project()
unit = project.length_unit
to_unit = 1 / store.KM_PER_MILE if unit == "mi" else 1.0

ui.page_header("Route Planning",
               "Generate alternative corridors, measure them against constraint layers, and compare the "
               "trade-offs. The tool does not choose a route.")
ui.synthetic_banner()

st.markdown(f"**{project.name}** · {project.start_name} → {project.end_name} · straight line "
            f"{project.straight_line_km * to_unit:,.1f} {unit}")
st.page_link("app_pages/1_Project_Setup.py", label="Change project or locations", icon=":material/edit_note:")

# --- Options ----------------------------------------------------------------
ss.setdefault("pg_route_strategies", [s.key for s in STRATEGIES])
ss.setdefault("pg_avoid_hard", True)
c1, c2, c3 = st.columns([3, 1.3, 1.3])
c1.multiselect("Routing strategies", [s.key for s in STRATEGIES], key="pg_route_strategies",
               format_func=lambda k: next(s.name for s in STRATEGIES if s.key == k))
c2.toggle("Treat protected areas as impassable", key="pg_avoid_hard",
          help="Hard constraints. Turning this off shows what a constraint-crossing route would look like; "
               "such routes are flagged and never ranked.")
c3.selectbox("Basemap", list(BASEMAPS), key="pg_basemap")

if not ss["pg_route_strategies"]:
    st.warning("Choose at least one routing strategy.")
    st.stop()

ui.require_valid_project(project)
plan = ui.plan_routes(project, ss["pg_route_strategies"], ss["pg_avoid_hard"])
routes = plan.routes
for m in plan.messages:
    st.info(m, icon=":material/info:")

# --- Map ----------------------------------------------------------------------
colors = route_colors(routes["route_id"].tolist())
highlight = st.radio("Highlight route", ["None"] + routes["route_id"].tolist(), horizontal=True,
                     format_func=lambda r: r if r == "None" else routes.set_index("route_id").loc[r, "name"])
st_folium(route_map(project, plan.layers, routes, plan.lines, ss["pg_basemap"],
                    None if highlight == "None" else highlight),
          height=560, use_container_width=True, returned_objects=[], key="route_map")
st.caption("Dashed outlines = SYNTHETIC demonstration features. Grey dashed line = straight-line reference "
           "(ignores constraints). Use the layer button (top right of the map) to toggle layers and routes.")

# --- Scoring ------------------------------------------------------------------
with st.expander("Criteria weights: how much each criterion matters to you", icon=":material/balance:"):
    st.caption("0 = ignore, 5 = very important. Scores are relative to the alternatives shown.")
    cols = st.columns(3)
    for i, c in enumerate(CRITERIA):
        key = f"pg_w_{c.key}"
        ss.setdefault(key, c.default_weight)
        cols[i % 3].slider(f"{c.label} ({c.category})", 0.0, 5.0, step=0.5, key=key,
                           help=("Higher is better" if not c.lower_is_better else "Lower is better") + f" · unit: {c.unit}")
    st.info("Social, Indigenous and cultural considerations are deliberately **not** weighted. They are not "
            "costs to be traded off, and they require authorised information from engagement (Phase 4).",
            icon=":material/diversity_3:")

weights = {c.key: ss.get(f"pg_w_{c.key}", c.default_weight) for c in CRITERIA}
scored = score_routes(routes, weights)

a, b = st.columns([2, 3])
a.plotly_chart(route_score_chart(scored), width="stretch")
b.plotly_chart(route_tradeoff_heatmap(scored), width="stretch")
terrain_label = "synthetic terrain" if plan.elevation.is_synthetic else "DEM: " + plan.elevation.label
st.plotly_chart(elevation_profile_chart(plan.profiles, scored.set_index("route_id")["name"].to_dict(), colors, unit,
                                        terrain_label),
                width="stretch")

# --- Comparison table ---------------------------------------------------------
st.subheader("Route comparison table")
table = scored[[
    "name", "comparison_status", "rank", "score", "length_km", "capex_cad", "lifecycle_pv_cad", "terrain_difficulty",
    "max_slope_pct", "water_crossings", "wetland_km", "forest_clearing_km", "new_access_km",
    "shared_corridor_km", "protected_km", "hard_constraints", "geohazards", "social_indigenous", "data_quality",
]].copy()
for col in ["length_km", "wetland_km", "forest_clearing_km", "new_access_km", "shared_corridor_km", "protected_km"]:
    table[col] = table[col] * to_unit
for col in ["capex_cad", "lifecycle_pv_cad"]:
    table[col] = store.to_currency(table[col], project) / 1e6
table["rank"] = table["rank"].map(lambda r: "—" if pd.isna(r) else str(int(r)))
st.dataframe(
    table, hide_index=True, width="stretch",
    column_config={
        "name": "Route", "comparison_status": "Status",
        "rank": st.column_config.TextColumn("Rank", help="Only routes without unresolved hard constraints are ranked."),
        "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
        "length_km": st.column_config.NumberColumn(f"Length ({unit})", format="%.1f"),
        "capex_cad": st.column_config.NumberColumn(f"CAPEX ({project.currency} M)*", format="%.0f"),
        "lifecycle_pv_cad": st.column_config.NumberColumn(f"Lifecycle PV ({project.currency} M)*", format="%.0f"),
        "terrain_difficulty": st.column_config.NumberColumn("Terrain (1-5)", format="%.1f"),
        "max_slope_pct": st.column_config.NumberColumn("Max grade (%)", format="%.1f"),
        "water_crossings": "Water crossings",
        "wetland_km": st.column_config.NumberColumn(f"Wetland ({unit})", format="%.1f"),
        "forest_clearing_km": st.column_config.NumberColumn(f"Forest clearing ({unit})", format="%.1f"),
        "new_access_km": st.column_config.NumberColumn(f"New access est. ({unit})", format="%.1f"),
        "shared_corridor_km": st.column_config.NumberColumn(f"Shared corridor ({unit})", format="%.1f"),
        "protected_km": st.column_config.NumberColumn(f"In protected area ({unit})", format="%.1f"),
        "hard_constraints": "Hard constraints", "geohazards": "Construction challenges",
        "social_indigenous": "Social & Indigenous", "data_quality": "Data quality",
    },
)
fx_note = "" if project.currency == "CAD" else f" Converted at your rate 1 CAD = {project.fx_rate_from_cad} {project.currency}."
st.caption(f"*{ESTIMATE_DISCLAIMER}{fx_note}")
st.download_button("Download comparison (CSV)", scored.drop(columns=[c for c in scored if c.startswith("norm_")])
                   .to_csv(index=False).encode(), "route_comparison.csv", "text/csv", icon=":material/download:")

# --- Trade-offs ---------------------------------------------------------------
st.subheader("Trade-offs in plain language")
notes = explain_tradeoffs(scored)
cols = st.columns(2)
for i, (_, r) in enumerate(scored.iterrows()):
    with cols[i % 2].container(border=True):
        rank = "not ranked" if r["has_hard_constraint"] else f"rank {int(r['rank'])} of comparable routes"
        st.markdown(f"<span style='color:{colors[r['route_id']]}'>■</span> **{r['name']}** · score "
                    f"{r['score']:.0f} · {rank}", unsafe_allow_html=True)
        st.caption(r["description"])
        st.markdown("\n".join(f"- {line}" for line in notes[r["route_id"]]))

c1, c2, c3, c4 = st.columns(4)
c1.markdown("**⛔ Hard constraints**  \nMust be avoided or formally resolved. Never offset by cost savings.")
c2.markdown("**⚠️ Avoidable impacts**  \nWetlands, forest clearing, water crossings: reducible by routing or mitigation.")
c3.markdown("**💲 Cost considerations**  \nIllustrative estimates; edit assumptions on Cost Optimizer.")
c4.markdown("**❓ Unknowns**  \nNot yet assessed. Missing data is never treated as 'no impact'.")

# --- Data sources & GIS layers --------------------------------------------------
st.divider()
st.subheader("GIS data layers & sources")
st.dataframe(pd.DataFrame(plan.register), hide_index=True, width="stretch",
             column_config={"data_type": "Data type", "confidence": "Confidence"})

user_rows = ui.user_layers(project)
with st.expander("Add your own GIS layer", icon=":material/upload_file:", expanded=False):
    if project.id is None:
        st.warning("Save the project on Project Setup first. Uploaded layers are stored with a saved project.")
    else:
        st.caption("Use open data you are licensed to use (e.g. government open-data portals). A layer replaces "
                   "the synthetic layer for its category. Do not upload confidential or culturally sensitive "
                   "locations without authorisation.")
        with st.form("upload_layer", clear_on_submit=True):
            file = st.file_uploader("GeoJSON, GeoPackage or zipped Shapefile", type=["geojson", "json", "gpkg", "zip"])
            c1, c2 = st.columns(2)
            category = c1.selectbox("Category", list(CATEGORIES), format_func=lambda k: CATEGORIES[k]["label"])
            layer_name = c2.text_input("Layer name")
            source = c1.text_input("Source / provider (required)")
            licence = c2.text_input("Licence")
            notes_text = st.text_input("Notes (date, scale, known gaps)")
            submitted = st.form_submit_button("Add layer", type="primary")
        if submitted:
            if file is None or not source.strip() or not layer_name.strip():
                st.error("Please provide a file, a layer name and its source.")
            else:
                try:
                    gdf, warnings = read_uploaded_layer(file.name, file.getvalue())
                    clipped, more = prepare_user_layer(gdf, category, corridor_frame(project), layer_name, source)
                    allowed = gdf[gdf.geometry.geom_type.isin(CATEGORIES[category]["geom"])]
                    for w in warnings + more:
                        st.warning(w)
                    if clipped.empty:
                        st.error("Nothing to add: no usable features inside the study area.")
                    else:
                        store.save_layer(project.id, category, layer_name.strip(), source.strip(),
                                         allowed.to_json(), licence.strip(), notes_text.strip())
                        st.toast(f"Added '{layer_name}' ({len(clipped)} features in study area)")
                        st.rerun()
                except Exception as exc:  # show any reading problem to the user instead of crashing
                    st.error(f"Could not read the file: {exc}")
    if user_rows:
        st.markdown("**Your layers**")
        options = {r["id"]: f"{CATEGORIES[r['category']]['label']}: {r['name']} ({r['source']})" for r in user_rows}
        to_remove = st.selectbox("Remove a layer", list(options), format_func=options.get, index=None)
        if st.button("Remove layer", disabled=to_remove is None):
            store.delete_layer(to_remove)
            st.rerun()

with st.expander("Elevation model (DEM)", icon=":material/terrain:"):
    dem = get_dem(project.id)
    if project.id is None:
        st.warning("Save the project on Project Setup first. The elevation model is stored with a saved project.")
    else:
        if dem:
            st.success(f"Using **{dem['name']}** (source: {dem['source']}, loaded {dem['loaded_at'].replace('T', ' ')}). "
                       f"Coverage of the study area: {plan.elevation.coverage:.0%}.", icon=":material/terrain:")
            if st.button("Remove elevation model (back to synthetic terrain)"):
                delete_dem(project.id)
                st.rerun()
        st.caption("Upload a GeoTIFF digital elevation model covering both ends of the route, from a free open "
                   "source such as Natural Resources Canada's elevation data (open.canada.ca) or Copernicus DEM "
                   "GLO-30. It is cropped to this project's study area and replaces the synthetic terrain for "
                   "slopes, profiles, environmental proxies and the 3D twin. One model per project.")
        with st.form("dem_upload", clear_on_submit=True):
            dem_file = st.file_uploader("GeoTIFF (.tif)", type=["tif", "tiff"])
            c1, c2 = st.columns(2)
            dem_source = c1.text_input("Source / provider (required)")
            dem_licence = c2.text_input("Licence")
            dem_notes = st.text_input("Notes (resolution, vertical datum, date)")
            if st.form_submit_button("Use this elevation model", type="primary"):
                if dem_file is None or not dem_source.strip():
                    st.error("Please choose a GeoTIFF and enter its source.")
                else:
                    try:
                        with st.spinner("Cropping and storing the elevation model…"):
                            save_dem(project.id, corridor_frame(project), dem_file.name, dem_file.getvalue(),
                                     dem_source.strip(), dem_licence.strip(), dem_notes.strip())
                        st.toast("Elevation model loaded", icon=":material/terrain:")
                        st.rerun()
                    except Exception as exc:  # show reading problems instead of crashing
                        st.error(f"Could not use this file: {exc}")

with st.expander("How routes are generated and measured", icon=":material/help:"):
    st.markdown(
        "1. The study area (a rectangle around the start-destination line) is divided into a grid.\n"
        "2. Each cell gets a cost from slope, water, wetland, forest and proximity to existing corridors, "
        "weighted differently for each strategy. Protected areas are impassable when that option is on.\n"
        "3. Dijkstra's shortest-path algorithm finds the least-cost corridor for each strategy.\n"
        "4. Each corridor is measured against the layers using exact geometry (overlap lengths, crossings, "
        "slopes from the elevation profile).\n\n"
        "Corridors are screening-level and indicative, not engineering alignments.")
    st.dataframe(pd.DataFrame([vars(s) for s in STRATEGIES]), hide_index=True, width="stretch")
    st.dataframe(pd.DataFrame(MEASUREMENT_ASSUMPTIONS.items(), columns=["Measurement assumption", "Value"]),
                 hide_index=True)
