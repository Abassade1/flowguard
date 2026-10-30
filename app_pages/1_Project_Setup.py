"""Project Setup (Module A): define, save and reopen pipeline projects."""

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from modules import projects as store
from modules import ui
from modules.config import PIPELINE_PROFILES
from modules.maps import BASEMAPS, project_map
from modules.projects import (
    ASSUMPTION_STATUSES, CAPACITY_UNITS, CURRENCIES, LENGTH_UNITS, Project, validate_project,
)

ss = st.session_state
FORM_FIELDS = [
    "name", "pipeline_type", "description", "start_name", "start_lat", "start_lon", "end_name", "end_lat",
    "end_lon", "approx_length_km", "diameter_mm", "capacity", "capacity_unit", "budget", "currency",
    "fx_rate_from_cad", "length_unit",
]
ASSUMPTION_COLUMNS = ["assumption", "status", "source"]

ui.page_header("Project Setup", "Define what is being evaluated, where, and under which assumptions.")

# --- Project library --------------------------------------------------------
saved = store.list_projects()
with st.container(border=True):
    c1, c2, c3, c4 = st.columns([3, 1, 1, 1], vertical_alignment="bottom")
    options = {p["id"]: f"{p['name']}  ·  saved {p['updated_at'].replace('T', ' ')}" for p in saved}
    choice = c1.selectbox("Saved projects", list(options), format_func=options.get, index=None,
                          placeholder="No saved projects yet" if not saved else "Choose a project to open…")
    if c2.button("Open", disabled=choice is None, width="stretch"):
        ui.set_active_project(store.load_project(choice), saved=True)
        st.rerun()
    if c3.button("New project", width="stretch"):
        ui.set_active_project(Project(), saved=False)
        st.rerun()
    if c4.button("Load demo", width="stretch", help="Synthetic demonstration project"):
        ui.set_active_project(store.load_sample_project(), saved=False)
        st.rerun()
    if choice is not None:
        with st.popover("Delete selected project…", icon=":material/delete:"):
            st.write(f"Permanently delete **{options[choice].split('  ·')[0]}** and its uploaded GIS layers?")
            if st.button("Yes, delete permanently", type="primary"):
                store.delete_project(choice)
                if ui.get_active_project().id == choice:
                    ui.set_active_project(store.load_sample_project(), saved=False)
                st.rerun()

# --- Load the active project into the form when it changes ------------------
project = ui.get_active_project()
if ss.get("pjf_rev") != ss["pg_project_rev"] or "pjf_name" not in ss:
    for f in FORM_FIELDS:
        value = getattr(project, f)
        ss[f"pjf_{f}"] = float(value) if isinstance(value, (int, float)) else value
    ss["pjf_assumptions_base"] = pd.DataFrame(project.assumptions, columns=ASSUMPTION_COLUMNS)
    ss["pjf_rev"] = ss["pg_project_rev"]

status = "Saved" if not ui.has_unsaved_changes() else ("Not saved yet" if project.id is None else "Unsaved changes")
st.markdown(f"#### {project.name} &nbsp;<span style='opacity:0.65;font-size:0.85rem'>· {status}"
            f"{'' if project.id is None else f' · ID {project.id}'}</span>", unsafe_allow_html=True)

# --- Locations ------------------------------------------------------------
# A map click is stored as "pending" and applied here, before the coordinate inputs are drawn
# (Streamlit does not allow changing a widget's value after it has been drawn in the same run).
pending = ss.pop("pjf_pending_point", None)
if pending:
    prefix = "pjf_start" if pending["target"] == "Start" else "pjf_end"
    ss[f"{prefix}_lat"], ss[f"{prefix}_lon"] = pending["lat"], pending["lon"]

st.subheader("1 · Start and destination")
map_col, input_col = st.columns([3, 2])
with input_col:
    target = st.radio("A click on the map sets the…", ["Start", "Destination"], horizontal=True,
                      key="pjf_click_target")
    st.text_input("Start name", key="pjf_start_name")
    a, b = st.columns(2)
    a.number_input("Start latitude", -90.0, 90.0, step=0.01, format="%.5f", key="pjf_start_lat")
    b.number_input("Start longitude", -180.0, 180.0, step=0.01, format="%.5f", key="pjf_start_lon")
    st.text_input("Destination name", key="pjf_end_name")
    a, b = st.columns(2)
    a.number_input("Destination latitude", -90.0, 90.0, step=0.01, format="%.5f", key="pjf_end_lat")
    b.number_input("Destination longitude", -180.0, 180.0, step=0.01, format="%.5f", key="pjf_end_lon")
    basemap = st.selectbox("Basemap", list(BASEMAPS), key="pg_basemap")

draft_for_map = Project.from_dict({**project.to_dict(), **{f: ss[f"pjf_{f}"] for f in FORM_FIELDS}})
with map_col:
    out = st_folium(project_map(draft_for_map, basemap), height=430, use_container_width=True,
                    returned_objects=["last_clicked"], key="setup_map")
    st.caption("Coordinates are WGS84 latitude/longitude. Locations are entered by you; this app does not "
               "look up place names online.")
click = (out or {}).get("last_clicked")
if click and click != ss.get("pjf_last_click"):
    ss["pjf_last_click"] = click
    ss["pjf_pending_point"] = {"target": target, "lat": round(click["lat"], 5), "lon": round(click["lng"], 5)}
    st.rerun()

# --- Project details -------------------------------------------------------
st.subheader("2 · Project details")
c1, c2 = st.columns(2)
c1.text_input("Project name", key="pjf_name")
c2.selectbox("Pipeline type", list(PIPELINE_PROFILES), key="pjf_pipeline_type",
             help="'Other' configurable types are planned for a later phase.")
st.text_area("Project description", key="pjf_description", height=90)

c1, c2, c3 = st.columns(3)
c1.number_input("Approximate pipeline length (km)", 0.0, 5000.0, step=5.0, key="pjf_approx_length_km",
                help="Your estimate. Route Planning measures each generated corridor.")
c1.caption(f"Straight-line distance: **{draft_for_map.straight_line_km:,.1f} km**")
c2.number_input("Pipeline diameter (mm)", 0.0, 3000.0, step=10.0, key="pjf_diameter_mm",
                help=f"{ss['pjf_diameter_mm'] / 25.4:,.1f} inches")
c2.caption(f"≈ {ss['pjf_diameter_mm'] / 25.4:,.1f} inches")
c3.number_input("Transport capacity", 0.0, 1e9, step=1000.0, key="pjf_capacity")
c3.selectbox("Capacity unit", CAPACITY_UNITS, key="pjf_capacity_unit")

st.subheader("3 · Budget, units and currency")
c1, c2, c3, c4 = st.columns(4)
c2.selectbox("Currency", CURRENCIES, key="pjf_currency")
c1.number_input(f"Proposed construction budget ({ss['pjf_currency']})", 0.0, 1e12, step=1e6, format="%.0f",
                key="pjf_budget")
c1.caption(store.fmt_money(ss["pjf_budget"], ss["pjf_currency"]))
c3.number_input(f"Exchange rate: 1 CAD = ? {ss['pjf_currency']}", 0.0001, 1000.0, step=0.01, format="%.4f",
                key="pjf_fx_rate_from_cad", disabled=ss["pjf_currency"] == "CAD",
                help="Enter a rate from a source you trust. No live rates are downloaded.")
c4.selectbox("Length unit for display", LENGTH_UNITS, key="pjf_length_unit")

# --- Assumptions -----------------------------------------------------------
st.subheader("4 · Planning assumptions")
st.caption("Record every assumption and how reliable it is. Reports will separate verified data, assumptions, "
           "estimates and synthetic data.")
edited = st.data_editor(
    ss["pjf_assumptions_base"], num_rows="dynamic", width="stretch", hide_index=True,
    key=f"pjf_assumptions_editor_{ss['pjf_rev']}",
    column_config={
        "assumption": st.column_config.TextColumn("Assumption", width="large", required=True),
        "status": st.column_config.SelectboxColumn("Status", options=ASSUMPTION_STATUSES, required=True,
                                                   default="Assumed"),
        "source": st.column_config.TextColumn("Source / reference"),
    },
)

# --- Build the working copy, validate, save --------------------------------
values = {f: ss[f"pjf_{f}"] for f in FORM_FIELDS}
if values["currency"] == "CAD":
    values["fx_rate_from_cad"] = 1.0
values["assumptions"] = [
    {k: ("" if pd.isna(v) else v) for k, v in row.items()}
    for row in edited.dropna(subset=["assumption"]).to_dict("records")
]
draft = Project.from_dict({**project.to_dict(), **values})
ui.update_active_project(draft)

problems = validate_project(draft)
for p in problems:
    st.error(p, icon=":material/error:")

c1, c2 = st.columns([1, 4], vertical_alignment="center")
if c1.button("Save project", type="primary", disabled=bool(problems), icon=":material/save:"):
    saved_project = store.save_project(draft)
    ui.set_active_project(saved_project, saved=True)
    st.toast(f"Saved '{saved_project.name}' (ID {saved_project.id})", icon=":material/check_circle:")
    st.rerun()
c2.caption("Saved to a local SQLite database: data/database/flowguard.db. Nothing leaves this computer.")
