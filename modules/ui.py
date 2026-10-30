"""Shared Streamlit building blocks: sidebar controls, headers, banners, caching.

This is the only file in `modules/` that imports Streamlit. Business logic lives
in the other modules so it can be tested without a running app.

Settings are kept in `st.session_state` under keys starting with "pg_" so they
survive when the user switches pages.
"""

import json
from dataclasses import dataclass, replace
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
import streamlit as st

from modules import projects as store
from modules.anomaly_detection import Thresholds, detect_events, events_as_of
from modules.ml_detection import MLSettings, run_ml
from modules import cost_uncertainty, risk_register
from modules.cost_engine import CostAssumptions, add_costs, estimate_route_cost, sensitivity
from modules.route_engine import CRITERIA, score_routes
from modules.elevation import elevation_for, get_dem, register_entry
from modules.environmental_engine import EnvThresholds, screen_routes
from modules.config import (
    APP_NAME, APP_VERSION, PIPELINE_PROFILES, STATUS_COLORS, SYNTHETIC_NOTICE,
)
from modules.gis_layers import assemble_layers, corridor_frame
from modules import social_impact
from modules.monitoring_engine import SCENARIOS, SimulationConfig, build_sections, simulate_readings
from modules.projects import Project
from modules.config import SOCIAL_NOT_ASSESSED
from modules.routing import STRATEGIES, elevation_profile, generate_alternatives, routes_to_wgs84

STATE_PREFIX = "pg_"

_SIM_DEFAULTS = SimulationConfig(scenario="Possible leak (flow imbalance)")
_THR_DEFAULTS = Thresholds()

DEFAULTS = {
    "pg_pipeline_type": _SIM_DEFAULTS.pipeline_type,
    "pg_length_km": int(_SIM_DEFAULTS.pipeline_length_km),
    "pg_n_sections": _SIM_DEFAULTS.n_sections,
    "pg_scenario": _SIM_DEFAULTS.scenario,
    "pg_affected_section": _SIM_DEFAULTS.affected_section,
    "pg_onset_hour": int(_SIM_DEFAULTS.onset_hour),
    "pg_severity": _SIM_DEFAULTS.severity,
    "pg_duration_hours": _SIM_DEFAULTS.duration_hours,
    "pg_noise_level": _SIM_DEFAULTS.noise_level,
    "pg_seed": _SIM_DEFAULTS.seed,
    "pg_link_project": True,
    "pg_ml_enabled": False,
    "pg_ml_model": "robust_covariance",
    "pg_ml_baseline": 24,
    "pg_ml_margin": 1.1,
    "pg_ml_min": 3,
    **{f"pg_thr_{name}": getattr(_THR_DEFAULTS, name) for name in Thresholds.__dataclass_fields__},
}


def init_state() -> None:
    """Fill in defaults, and stop Streamlit discarding settings from other pages.

    Streamlit forgets a widget's value when the page showing it is not displayed.
    Re-assigning each key to itself on every run keeps the value alive.
    """
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "pg_active_project" not in st.session_state:
        set_active_project(store.load_sample_project(), saved=False)
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith(STATE_PREFIX):
            st.session_state[key] = st.session_state[key]


def inject_css() -> None:
    st.markdown(
        """
        <style>
          .block-container {padding-top: 3.5rem; max-width: 1400px;}
          div[data-testid="stMetric"] {
              background: var(--secondary-background-color);
              border: 1px solid rgba(128,128,128,0.18); border-radius: 8px; padding: 0.7rem 0.9rem;
          }
          div[data-testid="stMetricLabel"] p {font-size: 0.8rem; opacity: 0.8;}
          .pg-badge {display:inline-block; padding:2px 10px; border-radius:999px; color:#fff;
                     font-size:0.78rem; font-weight:600;}
          .pg-kicker {text-transform: uppercase; letter-spacing: 0.08em; font-size: 0.72rem;
                      opacity: 0.7; margin-bottom: -0.6rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, kicker: str = APP_NAME) -> None:
    st.markdown(f"<div class='pg-kicker'>{kicker}</div>", unsafe_allow_html=True)
    st.title(title)
    st.caption(subtitle)


def synthetic_banner() -> None:
    st.info(SYNTHETIC_NOTICE, icon=":material/science:")


def status_badge(status: str) -> str:
    """HTML badge. The status text is always shown, so colour is never the only cue."""
    return f"<span class='pg-badge' style='background:{STATUS_COLORS[status]}'>{status}</span>"


def sidebar_controls() -> None:
    """Simulation scenario controls, shown in the sidebar on every page."""
    ss = st.session_state
    with st.sidebar:
        st.caption(f"Version {APP_VERSION}")  # the product logo above comes from st.logo in app.py
        project = get_active_project()
        st.markdown(f"**Active project**  \n{project.name}")
        if has_unsaved_changes():
            st.caption("Unsaved changes. Save them on Project Setup.")
        st.divider()
        st.markdown("**Simulation scenario**")
        st.toggle("Model the active project's route", key="pg_link_project",
                  help="Use the project's pipeline type and the length of its modelled route "
                       "(chosen on 3D Digital Twin).")
        linked = ss["pg_link_project"] and pipeline_route(project) is not None
        if linked:
            route = pipeline_route(project)
            st.caption(f"{project.pipeline_type} · {route['name']} · {route['length_km']:,.1f} km")
        st.selectbox("Pipeline type", list(PIPELINE_PROFILES), key="pg_pipeline_type", disabled=linked)
        st.selectbox("Scenario", list(SCENARIOS), key="pg_scenario", help="Which synthetic event to inject.")
        st.caption(SCENARIOS[ss["pg_scenario"]])

        # Keep dependent values inside their valid range before drawing the widgets.
        ss["pg_affected_section"] = min(ss["pg_affected_section"], ss["pg_n_sections"])
        ss["pg_onset_hour"] = min(ss["pg_onset_hour"], ss["pg_duration_hours"])

        disabled = ss["pg_scenario"] == "Normal operations"
        st.slider("Affected section", 1, ss["pg_n_sections"], key="pg_affected_section", disabled=disabled)
        st.slider("Event onset (hour of window)", 0, ss["pg_duration_hours"], key="pg_onset_hour",
                  disabled=disabled)
        st.slider("Event severity", 0.1, 1.0, step=0.1, key="pg_severity", disabled=disabled,
                  help="0.1 = subtle (may go undetected), 1.0 = severe.")

        with st.expander("Pipeline & simulation settings"):
            st.slider("Pipeline length (km)", 20, 500, step=10, key="pg_length_km", disabled=linked)
            st.slider("Number of sections", 4, 16, key="pg_n_sections")
            st.slider("Time window (hours)", 24, 168, step=12, key="pg_duration_hours")
            st.slider("Sensor noise multiplier", 0.0, 3.0, step=0.25, key="pg_noise_level")
            st.number_input("Random seed", 0, 9999, key="pg_seed",
                            help="Same seed + same settings = identical data.")


def get_sim_config() -> SimulationConfig:
    ss = st.session_state
    pipeline_type, length = ss["pg_pipeline_type"], float(ss["pg_length_km"])
    if ss.get("pg_link_project", True):
        project = get_active_project()
        route = pipeline_route(project)
        if route is not None:
            pipeline_type, length = project.pipeline_type, round(float(route["length_km"]), 1)
    return SimulationConfig(
        pipeline_type=pipeline_type,
        pipeline_length_km=length,
        n_sections=int(ss["pg_n_sections"]),
        duration_hours=int(ss["pg_duration_hours"]),
        scenario=ss["pg_scenario"],
        affected_section=int(min(ss["pg_affected_section"], ss["pg_n_sections"])),
        onset_hour=float(ss["pg_onset_hour"]),
        severity=float(ss["pg_severity"]),
        noise_level=float(ss["pg_noise_level"]),
        seed=int(ss["pg_seed"]),
    )


def get_ml_settings() -> MLSettings:
    ss = st.session_state
    return MLSettings(model=ss["pg_ml_model"], baseline_hours=float(ss["pg_ml_baseline"]),
                      threshold_margin=float(ss["pg_ml_margin"]), min_consecutive=int(ss["pg_ml_min"]))


def get_thresholds() -> Thresholds:
    ss = st.session_state
    return Thresholds(**{name: ss[f"pg_thr_{name}"] for name in Thresholds.__dataclass_fields__})


@st.cache_data(show_spinner="Training the experimental ML detector…")
def run_ml_detection(config: SimulationConfig, settings: MLSettings):
    """Train/score the optional ML detector (cached per simulation + ML settings)."""
    sections = build_sections(config)
    readings = simulate_readings(config)
    return run_ml(readings, sections, config.profile.nominal_temperature_c, settings)


@st.cache_data(show_spinner=False)
def rule_events_as_of(config: SimulationConfig, thresholds: Thresholds, timestamp: pd.Timestamp) -> pd.DataFrame:
    """Rule events exactly as known at `timestamp` (no look-ahead). Cached per replay moment."""
    sections = build_sections(config)
    readings = simulate_readings(config)
    return events_as_of(readings, sections, thresholds, config.profile.nominal_temperature_c, timestamp)


@st.cache_data(show_spinner="Simulating sensor readings…")
def run_monitoring(config: SimulationConfig, thresholds: Thresholds):
    """Simulate readings and detect events. Cached: identical settings reuse results."""
    sections = build_sections(config)
    readings = simulate_readings(config)
    events = detect_events(readings, sections, thresholds, config.profile.nominal_temperature_c)
    return sections, readings, events


# ---------------------------------------------------------------------------
# Active project
# ---------------------------------------------------------------------------
def get_active_project() -> Project:
    return Project.from_dict(st.session_state["pg_active_project"])


def set_active_project(project: Project, saved: bool) -> None:
    """Make `project` the working copy. `saved=True` means it matches the database."""
    data = project.to_dict()
    st.session_state["pg_active_project"] = data
    st.session_state["pg_project_saved"] = json.dumps(data, sort_keys=True) if saved else ""
    st.session_state["pg_project_rev"] = st.session_state.get("pg_project_rev", 0) + 1


def update_active_project(project: Project) -> None:
    """Store edits to the working copy without resetting the editing form."""
    st.session_state["pg_active_project"] = project.to_dict()


def has_unsaved_changes() -> bool:
    current = json.dumps(st.session_state["pg_active_project"], sort_keys=True)
    return current != st.session_state.get("pg_project_saved", "")


# ---------------------------------------------------------------------------
# Route planning (cached: the same project + layers + options reuse results)
# ---------------------------------------------------------------------------
@st.cache_resource
def _init_database() -> bool:
    store.init_db()
    return True


def user_layers(project: Project) -> list[dict]:
    _init_database()
    return store.list_layers(project.id)


@dataclass
class RoutePlan:
    routes: pd.DataFrame  # one row per alternative, with measured attributes
    lines: dict  # route_id -> [(lat, lon), ...]
    layers: dict[str, gpd.GeoDataFrame]  # constraint layers in EPSG:4326
    register: list[dict]  # data-source register
    profiles: dict[str, pd.DataFrame]  # route_id -> elevation profile
    messages: list[str]
    geoms_projected: dict  # route_id -> LineString in the project's UTM CRS (metres)
    layers_projected: dict[str, gpd.GeoDataFrame]  # constraint layers in UTM
    crs_epsg: int
    elevation: object  # SyntheticElevation or DemElevation (see modules/elevation.py)


@st.cache_data(show_spinner="Generating route corridors…")
def _plan_routes_cached(project_json: str, layer_ids: tuple, dem_id, strategy_keys: tuple, avoid_hard: bool):
    project = Project.from_dict(json.loads(project_json))
    frame = corridor_frame(project)
    layer_rows = store.list_layers(project.id) if layer_ids else []
    layers, register = assemble_layers(frame, layer_rows)
    elevation = elevation_for(frame, get_dem(project.id) if dem_id else None)
    register.append(register_entry(elevation))
    sources = ["user layers" if layer_rows else "synthetic layers",
               "synthetic terrain" if elevation.is_synthetic else "DEM terrain"]
    quality = "Synthetic demonstration" if sources == ["synthetic layers", "synthetic terrain"] else \
        "Mixed: " + " + ".join(sources)
    messages_pre = []
    if not elevation.is_synthetic and elevation.coverage < 0.95:
        messages_pre.append(f"The elevation model covers only {elevation.coverage:.0%} of the study area; "
                            "slopes are unknown elsewhere.")
    routes, geoms, messages = generate_alternatives(frame, layers, list(strategy_keys), avoid_hard, quality,
                                                    elevation)
    messages = messages_pre + messages
    return RoutePlan(
        routes=routes,
        lines=routes_to_wgs84(geoms, frame.crs_epsg),
        layers={cat: gdf.to_crs(4326) for cat, gdf in layers.items()},
        register=register,
        profiles={rid: elevation_profile(frame, g, elevation) for rid, g in geoms.items()},
        messages=messages,
        geoms_projected=geoms,
        layers_projected=layers,
        crs_epsg=frame.crs_epsg,
        elevation=elevation,
    )


def plan_routes(project: Project, strategy_keys: list[str] | None = None, avoid_hard: bool = True) -> RoutePlan:
    """Generate (or reuse cached) route alternatives for a project."""
    keys = tuple(strategy_keys or [s.key for s in STRATEGIES])
    # Only the fields that affect routing go into the cache key.
    geo = {k: getattr(project, k) for k in ("id", "start_lat", "start_lon", "end_lat", "end_lon")}
    layer_ids = tuple(r["id"] for r in user_layers(project))
    dem = get_dem(project.id)
    plan = _plan_routes_cached(json.dumps(geo), layer_ids, dem["id"] if dem else None, keys, avoid_hard)
    # Costs are recalculated every time (fast), so edited cost assumptions apply immediately.
    routes = add_costs(plan.routes, project.diameter_mm, cost_assumptions(project))
    # Engagement information is attached as text for each route option. It is never scored.
    records = engagement_records(project)
    routes["social_indigenous"] = [social_impact.route_social_status(records, key, SOCIAL_NOT_ASSESSED)
                                   for key in routes["strategy"]]
    return replace(plan, routes=routes)


def pipeline_route(project: Project, plan: RoutePlan | None = None) -> pd.Series | None:
    """The route option the user chose to model (twin & monitoring); first option if none chosen yet.

    Returns None when the project's locations are not valid yet. The tool never picks a route on merit:
    the default is simply the first generated option, and the user can change it on 3D Digital Twin.
    """
    if store.validate_project(project):
        return None
    plan = plan or plan_routes(project)
    candidates = plan.routes[plan.routes["route_id"] != "REF"]
    if candidates.empty:
        candidates = plan.routes
    match = candidates[candidates["strategy"] == project.design_route_strategy]
    return (match if not match.empty else candidates).iloc[0]


def require_valid_project(project: Project) -> None:
    """Stop the page with a clear message if the project cannot be routed yet."""
    problems = store.validate_project(project)
    if problems:
        st.error("This project's details need fixing before routes can be generated: " + " ".join(problems),
                 icon=":material/error:")
        st.page_link("app_pages/1_Project_Setup.py", label="Go to Project Setup", icon=":material/edit_note:")
        st.stop()


def engagement_records(project: Project) -> pd.DataFrame:
    _init_database()
    return social_impact.list_records(project.id)


def screen_environment(plan: "RoutePlan", project: Project) -> pd.DataFrame:
    """Environmental screening of every route in a plan (fast; not cached)."""
    frame = corridor_frame(project)
    t = EnvThresholds(row_width_m=cost_assumptions(project).row_width_m)
    return screen_routes(frame, plan.routes, plan.geoms_projected, plan.layers_projected, t, plan.elevation)


def project_analysis(project: Project) -> SimpleNamespace:
    """Everything the risk register and investor reports need, from every module, in one place."""
    plan = plan_routes(project)
    weights = {c.key: st.session_state.get(f"pg_w_{c.key}", c.default_weight) for c in CRITERIA}
    scored = score_routes(plan.routes, weights)
    design = pipeline_route(project, plan)
    a = cost_assumptions(project)
    estimates = {r["route_id"]: estimate_route_cost(r, project.diameter_mm, a) for _, r in scored.iterrows()}
    records = engagement_records(project)
    return SimpleNamespace(
        plan=plan, routes=scored, design=design, assumptions=a, estimates=estimates,
        sensitivity=sensitivity(design, project.diameter_mm, a),
        screening=screen_environment(plan, project),
        records=records, engagement=social_impact.summarise(records),
        commitments=social_impact.export_view(records[records["record_type"] == "commitment"]),
        risks=risk_register.list_risks(project.id),
        mc={r["route_id"]: cost_uncertainty.simulate(r, project.diameter_mm, a,
                                                      cost_uncertainty.ranges_with_overrides(project.cost_uncertainty))
            for _, r in scored.iterrows() if r["route_id"] != "REF"},
    )


def cost_assumptions(project: Project) -> CostAssumptions:
    return CostAssumptions.from_overrides(project.cost_assumptions)
