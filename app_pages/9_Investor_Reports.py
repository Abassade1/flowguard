"""Reports & Export (Module H): investor dashboard and downloadable PDF / CSV reports."""

import re
from datetime import date

import pandas as pd
import streamlit as st

from modules import risk_register as rr
from modules import ui
from modules.charts import cost_breakdown_chart, risk_matrix_chart
from modules.cost_engine import assumption_table
from modules.environmental_engine import route_summary
from modules.projects import fmt_money, to_currency
from modules.reporting import DATA_CLASSES, REPORT_STATUS, build_report, to_csv_zip, to_pdf

project = ui.get_active_project()
ui.page_header("Reports & Export", "Investor decision-support dashboard and downloadable reports.")
st.info(REPORT_STATUS, icon=":material/info:")
ui.require_valid_project(project)

a = ui.project_analysis(project)
routes, design = a.routes, a.design
cur, fx = project.currency, to_currency(1.0, project)
comparable = routes[(routes["route_id"] != "REF") & ~routes["has_hard_constraint"]]
de = a.estimates[design["route_id"]]

# --- Dashboard ------------------------------------------------------------------------
st.subheader("Investor dashboard")
synthetic_in_use = any(r["data_type"] == "Synthetic demonstration" for r in a.plan.register)
if synthetic_in_use:
    st.warning("Synthetic demonstration data is in use (mapping layers and/or terrain). Results are for exploring "
               "the tool, not for investment decisions.", icon=":material/science:")

m = st.columns(5)
m[0].metric("Comparable route options", len(comparable), f"{len(routes) - 1 - len(comparable)} with hard constraints",
            delta_color="off", delta_arrow="off")
if not comparable.empty:
    capex = comparable["capex_cad"] * fx
    m[1].metric(f"CAPEX range ({cur})", f"{capex.min() / 1e6:,.0f}–{capex.max() / 1e6:,.0f} M", "estimate",
                delta_color="off", delta_arrow="off")
m[2].metric(f"Modelled route lifecycle PV ({cur})", f"{de.lifecycle_pv * fx / 1e6:,.0f} M", design["name"],
            delta_color="off", delta_arrow="off")
if project.budget:
    m[3].metric("Modelled route CAPEX vs budget", f"{de.capex * fx / project.budget:.0%}",
                f"budget {fmt_money(project.budget, cur)}", delta_color="off", delta_arrow="off")
else:
    m[3].metric("Modelled route CAPEX vs budget", "no budget")
open_risks = a.risks[a.risks["status"] != "Closed"]
m[4].metric("High / very high risks (residual)", int(open_risks["residual_level"].isin(["High", "Very high"]).sum()),
            f"{len(open_risks)} open", delta_color="off", delta_arrow="off")
st.caption(f"The modelled route ({design['name']}) is the team's choice on 3D Digital Twin; the tool does not "
           "select a route.")

left, right = st.columns([3, 2])
left.plotly_chart(cost_breakdown_chart({routes.set_index("route_id").loc[rid, "name"]: e.by_group()
                                        for rid, e in a.estimates.items() if rid != "REF"}, cur, fx), width="stretch")
right.plotly_chart(risk_matrix_chart(rr.matrix_counts(a.risks, residual=True), "Open risks (residual)"),
                   width="stretch")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Environmental screening (count of themes per level)**")
    st.dataframe(route_summary(a.screening[a.screening["route_id"] != "REF"]), hide_index=True, width="stretch")
with c2:
    st.markdown("**Social & community (records, not consent)**")
    e = a.engagement
    st.dataframe(pd.DataFrame({"Measure": ["Engagement activities", "Concerns not resolved", "Unresolved disagreements",
                                           "Overdue commitments & actions", "Project changes from engagement"],
                               "Count": [e["activities"], e["open_concerns"], e["unresolved"], e["overdue"],
                                         e["changes"]]}), hide_index=True, width="stretch")

gaps = sorted(a.screening.loc[a.screening["level"] == "Not assessed", "label"].unique())
st.markdown("**Key data gaps and uncertainty**")
st.markdown("\n".join(f"- {g}" for g in [
    *(f"Not assessed: {g}" for g in gaps),
    "Mapping layers and/or terrain are synthetic" if synthetic_in_use else None,
    f"Largest cost driver: {a.sensitivity['assumption'].iloc[0]}" if not a.sensitivity.empty else None,
    "Cultural heritage not assessed (authorised information required)",
    "No engagement recorded yet" if e["activities"] == 0 else None,
] if g))

# --- Report ---------------------------------------------------------------------------------
st.divider()
st.subheader("Download the report")
report = build_report(project, routes, design["route_id"], a.estimates, assumption_table(a.assumptions),
                      a.sensitivity, a.screening, a.plan.register, a.engagement, a.commitments, a.risks, a.mc)
slug = re.sub(r"[^a-z0-9]+", "_", project.name.lower()).strip("_") or "project"
c1, c2 = st.columns(2)
c1.download_button("Download PDF report", to_pdf(report), f"{slug}_report_{date.today():%Y%m%d}.pdf",
                   "application/pdf", icon=":material/picture_as_pdf:", type="primary", width="stretch")
c2.download_button("Download data (CSV bundle, zip)", to_csv_zip(report), f"{slug}_data_{date.today():%Y%m%d}.zip",
                   "application/zip", icon=":material/folder_zip:", width="stretch")
st.caption("Restricted engagement records are never included. Every table carries a data_class column; the zip "
           "includes a README explaining the classes.")

with st.expander("Data classes used in reports", icon=":material/label:"):
    st.dataframe(pd.DataFrame(DATA_CLASSES.items(), columns=["Class", "Meaning"]), hide_index=True, width="stretch")

st.markdown("#### Report preview")
for section in report.sections:
    with st.expander(f"{section.title}  ·  {', '.join(dict.fromkeys(section.classes))}"):
        for p in section.paragraphs:
            st.write(p)
        if section.bullets:
            st.markdown("\n".join(f"- {b}" for b in section.bullets))
        for t in section.tables:
            st.caption(t.caption)
            st.dataframe(t.df, hide_index=True, width="stretch")
