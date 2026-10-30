"""Executive Overview: one-screen summary of operations and route planning."""

import streamlit as st

from modules import ui
from modules.anomaly_detection import events_at, section_status
from modules.charts import route_score_chart, status_strip
from modules.config import MONITORING_DISCLAIMER
from modules.projects import fmt_money, to_currency
from modules.route_engine import score_routes
from modules.risk_register import list_risks
from modules.social_impact import summarise

ui.page_header("Executive Overview", "Operational status and planning snapshot for the demonstration pipeline.")
ui.synthetic_banner()

config = ui.get_sim_config()
sections, readings, events = ui.run_monitoring(config, ui.get_thresholds())
latest_ts = readings["timestamp"].max()
current = events_at(events, latest_ts)
statuses = section_status(sections, current)

# --- Headline metrics -------------------------------------------------------
st.subheader("Operations (simulated)")
latest = readings[readings["timestamp"] == latest_ts]
alert_sections = int((statuses["status"] == "Alert").sum())
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Pipeline", f"{config.pipeline_length_km:,.0f} km", f"{config.pipeline_type}, {config.n_sections} sections",
          delta_color="off", delta_arrow="off")
c2.metric("Sections in Alert", alert_sections, f"of {config.n_sections}", delta_color="off", delta_arrow="off")
c3.metric("Active events", len(current), f"{len(events)} in last {config.duration_hours} h", delta_color="off", delta_arrow="off")
c4.metric("Inlet flow (latest)", f"{latest['flow_in'].iloc[0]:,.0f}", config.profile.flow_unit, delta_color="off", delta_arrow="off")
c5.metric("Lowest sensor health", f"{latest['sensor_health'].min():.0f}/100",
          latest.loc[latest["sensor_health"].idxmin(), "section_id"], delta_color="off", delta_arrow="off")

st.plotly_chart(status_strip(sections, statuses), width="stretch")

left, right = st.columns([3, 2])
with left:
    st.markdown("**Highest-priority active events**")
    if current.empty:
        st.success("No active events at the latest reading.", icon=":material/check_circle:")
    else:
        st.dataframe(
            current[["section_id", "severity", "rule", "start", "explanation"]].head(5),
            hide_index=True, width="stretch",
            column_config={"start": st.column_config.DatetimeColumn("Since", format="MMM D HH:mm"),
                           "explanation": st.column_config.TextColumn("Why", width="large")},
        )
    st.page_link("app_pages/7_Pipeline_Monitoring.py", label="Open FlowGuard Monitoring",
                 icon=":material/arrow_forward:")
with right:
    st.markdown("**Scenario**")
    st.write(f"{config.scenario}" + ("" if config.scenario == "Normal operations"
                                     else f" · section S{config.affected_section:02d} · severity {config.severity:.1f}"))
    st.caption(MONITORING_DISCLAIMER)

# --- Planning snapshot ------------------------------------------------------
st.divider()
project = ui.get_active_project()
st.subheader(f"Route planning snapshot: {project.name}")
ui.require_valid_project(project)
plan = ui.plan_routes(project)
scored = score_routes(plan.routes)
a, b = st.columns([3, 2])
with a:
    st.plotly_chart(route_score_chart(scored), width="stretch")
with b:
    capex = to_currency(scored.loc[scored["route_id"] != "REF", "capex_cad"], project)
    st.metric("Estimated CAPEX range", f"{fmt_money(capex.min(), project.currency)} – "
              f"{fmt_money(capex.max(), project.currency)}",
              "illustrative only", delta_color="off", delta_arrow="off")
    n_hard = int(scored["has_hard_constraint"].sum())
    st.metric("Alternatives with unresolved hard constraints", n_hard, f"of {len(scored)}", delta_color="off", delta_arrow="off")
    st.caption("Scores use default weights and compare these alternatives only. No route is selected "
               "automatically; social and Indigenous considerations are not yet assessed.")
    st.page_link("app_pages/2_Route_Planning.py", label="Open Route Planning", icon=":material/arrow_forward:")

# --- Environmental & community snapshot ---------------------------------------
st.divider()
st.subheader("Environmental, community & risk snapshot")
screening = ui.screen_environment(plan, project)
proposed = screening[screening["route_id"] != "REF"]
engagement = summarise(ui.engagement_records(project))
e1, e2, e3, e4 = st.columns(4)
e1.metric("Environmental data gaps", proposed.loc[proposed["level"] == "Not assessed", "label"].nunique(),
          "themes not assessed", delta_color="off", delta_arrow="off")
e2.metric("High screening flags", int((proposed["level"] == "High").sum()),
          f"across {proposed['route_id'].nunique()} route options", delta_color="off", delta_arrow="off")
e3.metric("Community concerns not resolved", engagement["open_concerns"],
          f"{engagement['activities']} engagement activities", delta_color="off", delta_arrow="off")
e4.metric("Overdue commitments & actions", engagement["overdue"], delta_color="off", delta_arrow="off")
st.caption("Screening flags are not an environmental assessment. Engagement records are not consent."
           + ("" if project.id else " Save the project to start recording engagement."))
risks = list_risks(project.id)
open_risks = risks[risks["status"] != "Closed"]
r1, r2, r3 = st.columns(3)
r1.metric("Open risks", len(open_risks))
r2.metric("High / very high (residual)", int(open_risks["residual_level"].isin(["High", "Very high"]).sum()))
r3.metric("Mitigations not started", int((open_risks["mitigation_status"] == "Not started").sum()))
l1, l2 = st.columns(2)
l1.page_link("app_pages/4_Environmental_Intelligence.py", label="Open Environmental Intelligence",
             icon=":material/arrow_forward:")
l2.page_link("app_pages/5_Community_Engagement.py", label="Open Indigenous & Community Engagement",
             icon=":material/arrow_forward:")
l1.page_link("app_pages/8_Risk_Register.py", label="Open Risk & Mitigation Register", icon=":material/arrow_forward:")
l2.page_link("app_pages/9_Investor_Reports.py", label="Open Reports & Export", icon=":material/arrow_forward:")
