"""3D Digital Twin (Module F): planning, construction and operations views on the project's terrain."""

import pandas as pd
import streamlit as st

from modules import ui
from modules.anomaly_detection import events_at, section_status
from modules.digital_twin import (
    CONSTRUCTION_STAGES, MODES, build_scene, construction_progress_pct, render_plotly, scene_to_json,
)
from modules.gis_layers import corridor_frame
from modules.maps import route_colors

ss = st.session_state
project = ui.get_active_project()

ui.page_header("3D Digital Twin",
               "One model from planning to operations: terrain, constraints, route options, construction "
               "progress and simulated sensor status.")
ui.require_valid_project(project)

plan = ui.plan_routes(project)
routes = plan.routes
names = routes.set_index("route_id")["name"].to_dict()
strategy_of = routes.set_index("route_id")["strategy"].to_dict()

# --- Controls -----------------------------------------------------------------------
c1, c2 = st.columns([1.2, 2])
mode = c1.segmented_control("Mode", MODES, default="Planning", key="pg_twin_mode") or "Planning"
options = [r for r in routes["route_id"] if r != "REF"] or ["REF"]
current = ui.pipeline_route(project, plan)["route_id"]
chosen = c2.selectbox("Route option to model as the pipeline", options, index=options.index(current),
                      format_func=names.get,
                      help="Your choice. The tool does not select a route. It sets the length used by "
                           "monitoring and the sections shown here.")
if strategy_of[chosen] != project.design_route_strategy:
    project.design_route_strategy = strategy_of[chosen]
    ui.update_active_project(project)
    st.rerun()  # sidebar & simulation pick up the new route length

with st.expander("View options", icon=":material/tune:"):
    v1, v2, v3, v4 = st.columns(4)
    show_terrain = v1.toggle("Terrain", value=True)
    show_features = v2.toggle("Constraint layers", value=True)
    show_alternatives = v3.toggle("Other route options", value=True, disabled=mode != "Planning")
    exaggeration = v4.slider("Vertical exaggeration", 1, 60, 15,
                             help="Pipelines are long and terrain is low: relief is exaggerated so it is visible.")

terrain_note = ("SYNTHETIC terrain (not real). Upload a GeoTIFF DEM on Route Planning to use real elevations."
                if plan.elevation.is_synthetic else f"Terrain: {plan.elevation.label}, "
                f"{plan.elevation.coverage:.0%} coverage.")
if plan.elevation.is_synthetic:
    ui.synthetic_banner()
st.caption(f"Real project coordinates (UTM EPSG:{plan.crs_epsg}). {terrain_note} Vertical scale ×{exaggeration}.")

# --- Data for the chosen mode -----------------------------------------------------------
config = ui.get_sim_config()
sections, readings, events = ui.run_monitoring(config, ui.get_thresholds())
section_ids = sections["section_id"].tolist()
if ss.get("pg_selected_section") not in section_ids:
    ss["pg_selected_section"] = section_ids[0]

statuses = snapshot = None
progress = {k: v for k, v in project.construction_progress.items() if k in section_ids}
s1, s2 = st.columns([1, 3])
s1.selectbox("Section", section_ids, key="pg_selected_section")
if mode == "Operations":
    timestamps = sorted(readings["timestamp"].unique())
    idx = s2.slider("Time (simulated)", 0, len(timestamps) - 1, len(timestamps) - 1, format="reading %d")
    now = pd.Timestamp(timestamps[idx])
    statuses = section_status(sections, events_at(ui.rule_events_as_of(config, ui.get_thresholds(), now), now))
    snapshot = readings[readings["timestamp"] == now]
    st.caption(f"Simulated readings at {now:%Y-%m-%d %H:%M} (scenario: {config.scenario}). Monitoring is a "
               "demonstration and cannot detect real leaks.")
elif mode == "Construction":
    s2.metric("Construction progress (length-weighted)", f"{construction_progress_pct(sections, progress):.0f}%")

scene = build_scene(corridor_frame(project), routes, plan.geoms_projected, plan.layers_projected, plan.elevation,
                    chosen, sections, route_colors(routes["route_id"].tolist()), project.pipeline_type)
selected = ss["pg_selected_section"]
st.plotly_chart(render_plotly(scene, mode, statuses, snapshot, progress, selected, show_terrain, show_features,
                              show_alternatives, exaggeration, config.profile.flow_unit), width="stretch")
st.caption("Drag to rotate, scroll to zoom, hover for details. Click legend items to hide them. "
           "Facilities are indicative placements, not designs.")

# --- Mode-specific panels -----------------------------------------------------------------
if mode == "Construction":
    st.subheader("Construction progress by section")
    st.caption("Record the current stage of each section. Saved with the project (Project Setup → Save).")
    table = pd.DataFrame({"section_id": section_ids,
                          "stage": [progress.get(s, CONSTRUCTION_STAGES[0]) for s in section_ids]})
    edited = st.data_editor(
        table, hide_index=True, width="stretch", disabled=["section_id"], key=f"progress_{ss['pg_project_rev']}",
        column_config={"section_id": "Section",
                       "stage": st.column_config.SelectboxColumn("Stage", options=CONSTRUCTION_STAGES, required=True)})
    new_progress = {r["section_id"]: r["stage"] for _, r in edited.iterrows() if r["stage"] != CONSTRUCTION_STAGES[0]}
    if new_progress != project.construction_progress:
        project.construction_progress = new_progress
        ui.update_active_project(project)
        st.rerun()

sec = scene.sections.set_index("section_id").loc[selected]
st.markdown(f"#### Section {selected}" + (f" &nbsp;{ui.status_badge(statuses.set_index('section_id').loc[selected, 'status'])}"
                                           if statuses is not None else ""), unsafe_allow_html=True)
d1, d2 = st.columns(2)
d1.dataframe(pd.DataFrame({
    "Property": ["Chainage", "Length", "Sensor station", "Sensor position", "Pipe elevation at sensor",
                 "Construction stage"],
    "Value": [f"KP {sec['start_km']:.1f} – {sec['end_km']:.1f} km", f"{sec['end_km'] - sec['start_km']:.1f} km",
              f"KP {sec['sensor_km']:.1f} km", f"{sec['lat']:.5f}, {sec['lon']:.5f}",
              "unknown (no elevation data)" if pd.isna(sec["z_m"]) else f"{sec['z_m']:,.0f} m",
              progress.get(selected, CONSTRUCTION_STAGES[0])],
}), hide_index=True, width="stretch")
if snapshot is not None:
    r = snapshot.set_index("section_id").loc[selected]
    d2.dataframe(pd.DataFrame({
        "Measurement (simulated)": ["Pressure", "Expected pressure", "Temperature", "Inlet / outlet flow",
                                    "Flow imbalance", "Sensor health"],
        "Value": ["no data" if pd.isna(r["pressure_kpa"]) else f"{r['pressure_kpa']:,.0f} kPa",
                  f"{sec['expected_pressure_kpa']:,.0f} kPa",
                  "no data" if pd.isna(r["temperature_c"]) else f"{r['temperature_c']:.1f} °C",
                  f"{r['flow_in']:,.0f} / {r['flow_out']:,.0f} {config.profile.flow_unit}",
                  f"{r['flow_imbalance_pct']:.2f} %", f"{r['sensor_health']:.0f}/100"],
    }), hide_index=True, width="stretch")
else:
    d2.caption("Switch to Operations mode to see simulated readings for this section.")

st.download_button("Export 3D scene (JSON)", scene_to_json(scene).encode(), "twin_scene.json", "application/json",
                   icon=":material/download:",
                   help="Terrain, layers, routes, sections and facilities in local coordinates, for use in "
                        "other 3D tools.")
