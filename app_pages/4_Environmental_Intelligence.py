"""Environmental Intelligence (Module D): screening support for route alternatives."""

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from modules import ui
from modules.environmental_engine import (
    ASSESSMENT_DISCLAIMER, LEVEL_COLORS, THEMES_BY_KEY, EnvThresholds, conflicts, level_matrix, route_summary,
)
from modules.gis_layers import CATEGORIES
from modules.maps import route_map

ss = st.session_state
project = ui.get_active_project()

ui.page_header("Environmental Intelligence",
               "Screen each route alternative for potential environmental conflicts, data gaps and "
               "mitigation ideas for professional review.")
st.warning(ASSESSMENT_DISCLAIMER, icon=":material/gavel:")
ui.synthetic_banner()

ui.require_valid_project(project)
plan = ui.plan_routes(project, ss.get("pg_route_strategies"), ss.get("pg_avoid_hard", True))
screening = ui.screen_environment(plan, project)
routes = plan.routes
names = routes.set_index("route_id")["name"].to_dict()

# --- Overview matrix ------------------------------------------------------------
st.subheader("Screening matrix")
st.caption("High / Moderate / Low are relative screening flags from simple rules. "
           "**Not assessed** means the data needed is missing; it does not mean low impact.")


LEVEL_ICONS = {"High": "🔴", "Moderate": "🟠", "Low": "🟢", "Not assessed": "⚪"}
matrix = level_matrix(screening).map(lambda level: f"{LEVEL_ICONS[level]} {level}")
matrix.index.name, matrix.columns.name = "Theme", None
st.dataframe(matrix, width="stretch", height=35 * (len(matrix) + 1) + 3)

summary = route_summary(screening)
st.dataframe(summary, hide_index=True, width="stretch",
             column_config={"route": "Route",
                            "Synthetic-only themes": st.column_config.NumberColumn(
                                "Themes based only on synthetic data",
                                help="Results with no real-world meaning until real data is loaded.")})

# --- Data gaps ----------------------------------------------------------------
gaps = screening[screening["level"] == "Not assessed"]["label"].unique()
synthetic = screening[screening["confidence"].str.startswith("None")]["label"].unique()
c1, c2 = st.columns(2)
with c1.container(border=True):
    st.markdown("**Data gaps (not assessed)**")
    st.markdown("\n".join(f"- {g}" for g in gaps) if len(gaps) else "None. Every theme has some data.")
with c2.container(border=True):
    st.markdown("**Themes relying on synthetic data (no real-world meaning)**")
    st.markdown("\n".join(f"- {g}" for g in synthetic) if len(synthetic) else "None.")
st.caption("Upload real layers on Route Planning → *Add your own GIS layer* to close gaps. A 'Sensitive "
           "wildlife habitat' layer enables the habitat & biodiversity theme.")

# --- One route in detail ----------------------------------------------------------
st.divider()
choices = [r for r in routes["route_id"] if r != "REF"] + ["REF"]
rid = st.selectbox("Route to examine", choices, format_func=names.get)
detail = screening[screening["route_id"] == rid]

left, right = st.columns([3, 2])
with left:
    st_folium(route_map(project, plan.layers, routes, plan.lines, ss.get("pg_basemap", "OpenStreetMap (online)"),
                        highlight=rid),
              height=460, use_container_width=True, returned_objects=[], key="env_map")
with right:
    st.markdown("**Features crossed or nearby**")
    found = conflicts(plan.geoms_projected[rid], plan.layers_projected, EnvThresholds())
    if found.empty:
        st.info("No mapped features crossed or within screening distance. Only as complete as the layers used.")
    else:
        found["category"] = found["category"].map(lambda c: CATEGORIES[c]["label"])
        st.dataframe(found, hide_index=True, width="stretch", height=420,
                     column_config={"km": st.column_config.NumberColumn("km inside", format="%.1f")})

st.markdown(f"#### Themes for {names[rid]}")
for _, row in detail.iterrows():
    theme = THEMES_BY_KEY[row["theme"]]
    color = LEVEL_COLORS[row["level"]]
    with st.container(border=True):
        st.markdown(f"<span style='display:inline-block;padding:1px 9px;border-radius:999px;background:{color};"
                    f"color:white;font-size:0.78rem;font-weight:600'>{row['level']}</span> &nbsp;"
                    f"**{row['label']}**", unsafe_allow_html=True)
        st.write(row["explanation"])
        st.caption(f"Data: {row['data_status']} · Confidence: {row['confidence']}")
        with st.expander("Mitigation ideas for professional review"):
            st.markdown("\n".join(f"- {m}" for m in theme.mitigation))

export = screening.drop(columns=["theme"]).rename(columns={"label": "theme"})
st.download_button("Download screening results (CSV)", export.to_csv(index=False).encode(),
                   "environmental_screening.csv", "text/csv", icon=":material/download:")

with st.expander("Screening rules and distances", icon=":material/help:"):
    st.dataframe(pd.DataFrame(vars(EnvThresholds()).items(), columns=["Setting", "Value"]), hide_index=True)
    st.markdown(
        "- **Wetlands:** High if > 1 km crossed or ≥ 2 wetlands; Moderate if any crossing or > 0.5 km nearby.\n"
        "- **Water:** High if > 5 crossings; Moderate if any.\n"
        "- **Protected areas:** High if crossed; Moderate if within the buffer.\n"
        "- **Forest:** High if > 20 km cleared; Moderate if any.\n"
        "- **Fragmentation:** High if ≥ 2 intact patches split or > 150 km new disturbance.\n"
        "- **Erosion / landslide / flood:** proxies from terrain grade and distance to water.\n"
        "- **Habitat & biodiversity:** only with uploaded habitat data; otherwise *Not assessed*.")
