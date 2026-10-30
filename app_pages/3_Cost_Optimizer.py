"""Cost Optimizer (Module C): editable assumptions, CAPEX/OPEX/lifecycle, breakdown and sensitivity."""

from dataclasses import fields

import pandas as pd
import streamlit as st

from modules import projects as store
from modules import ui
from modules.charts import cost_breakdown_chart, cumulative_cost_chart, tornado_chart
from modules.config import ESTIMATE_DISCLAIMER
from modules.cost_engine import (
    ASSUMPTION_GROUPS, SENSITIVITY_METRICS, CostAssumptions, assumption_table, cumulative_cost,
    estimate_route_cost, sensitivity,
)
from modules.maps import route_colors

ss = st.session_state
project = ui.get_active_project()
cur = project.currency
fx = store.to_currency(1.0, project)  # CAD -> project currency
unit = project.length_unit
to_unit = 1 / store.KM_PER_MILE if unit == "mi" else 1.0

ui.page_header("Cost Optimizer", "CAPEX, OPEX and lifecycle cost for each route alternative, with editable "
                                 "assumptions and sensitivity analysis.")
st.warning(ESTIMATE_DISCLAIMER, icon=":material/request_quote:")
st.markdown(f"**{project.name}** · diameter {project.diameter_mm:,.0f} mm ({project.diameter_mm / 25.4:.0f} in) · "
            f"budget {store.fmt_money(project.budget, cur)}"
            + ("" if cur == "CAD" else f" · shown in {cur} at your rate 1 CAD = {project.fx_rate_from_cad} {cur}"))

# --- Assumptions --------------------------------------------------------------
if ss.pop("cost_reset", False):
    project.cost_assumptions = {}
    ui.update_active_project(project)
    ss["cost_rev"] = None
if ss.get("cost_rev") != ss["pg_project_rev"] or "cost_steel_price_cad_per_t" not in ss:
    current = ui.cost_assumptions(project)
    for f in fields(CostAssumptions):
        ss[f"cost_{f.name}"] = float(getattr(current, f.name))
    ss["cost_rev"] = ss["pg_project_rev"]

with st.expander("Cost assumptions (illustrative defaults, editable)", icon=":material/tune:"):
    tabs = st.tabs(ASSUMPTION_GROUPS)
    for group, tab in zip(ASSUMPTION_GROUPS, tabs):
        group_fields = [f for f in fields(CostAssumptions) if f.metadata["group"] == group]
        cols = tab.columns(2)
        for i, f in enumerate(group_fields):
            m = f.metadata
            cols[i % 2].number_input(
                f"{m['label']} ({m['unit']})", min_value=0.0, step=float(m["step"]), key=f"cost_{f.name}",
                format="%.2f" if f.default < 100 else "%.0f", help=m["help"] or None,
            )
    assumptions = CostAssumptions(**{f.name: ss[f"cost_{f.name}"] for f in fields(CostAssumptions)})
    if assumptions.overrides() != project.cost_assumptions:
        project.cost_assumptions = assumptions.overrides()
        ui.update_active_project(project)

    register = assumption_table(assumptions)
    edited_count = int((register["status"] == "User-edited").sum())
    st.caption(f"{edited_count} assumption(s) edited from the illustrative defaults. Edits are stored with the "
               "project when you save it.")
    st.dataframe(register.drop(columns="key"), hide_index=True, width="stretch",
                 column_config={"value": st.column_config.NumberColumn("Value", format="%.4g"),
                                "default": st.column_config.NumberColumn("Default", format="%.4g")})
    c1, c2, _ = st.columns([1, 1, 3])
    if c1.button("Reset to defaults", disabled=edited_count == 0):
        ss["cost_reset"] = True
        st.rerun()
    if project.id is not None:
        if c2.button("Save project", type="primary", disabled=not ui.has_unsaved_changes(), icon=":material/save:"):
            ui.set_active_project(store.save_project(project), saved=True)
            st.toast("Project and cost assumptions saved", icon=":material/check_circle:")
            st.rerun()
    else:
        c2.caption("Save the project on Project Setup to keep these assumptions.")

# --- Route costs ---------------------------------------------------------------
ui.require_valid_project(project)
plan = ui.plan_routes(project, ss.get("pg_route_strategies"), ss.get("pg_avoid_hard", True))
routes = plan.routes
colors = route_colors(routes["route_id"].tolist())
names = routes.set_index("route_id")["name"].to_dict()
estimates = {r["route_id"]: estimate_route_cost(r, project.diameter_mm, assumptions) for _, r in routes.iterrows()}

st.subheader("Route cost summary")
summary = pd.DataFrame([{
    "Route": names[rid],
    "Status": "Reference only (ignores constraints)" if rid == "REF" else (
        "Unresolved hard constraint" if routes.set_index("route_id").loc[rid, "hard_constraints"] else "Comparable"),
    f"Length ({unit})": e.length_km * to_unit,
    f"CAPEX ({cur} M)": e.capex * fx / 1e6,
    f"CAPEX per {unit} ({cur} M)": e.capex_per_km * fx / 1e6 / to_unit,
    f"OPEX per year ({cur} M)": e.opex_annual * fx / 1e6,
    f"Lifecycle, undiscounted ({cur} M)": e.lifecycle_nominal * fx / 1e6,
    f"Lifecycle PV ({cur} M)": e.lifecycle_pv * fx / 1e6,
    "CAPEX vs budget": (e.capex * fx / project.budget) if project.budget else None,
} for rid, e in estimates.items()])
st.dataframe(summary, hide_index=True, width="stretch", column_config={
    **{c: st.column_config.NumberColumn(c, format="%.1f") for c in summary.columns if "(" in c},
    "CAPEX vs budget": st.column_config.ProgressColumn(
        "CAPEX vs budget", min_value=0.0, max_value=max(1.5, float(summary["CAPEX vs budget"].max() or 0)),
        format="percent", help="100% = equal to the proposed construction budget (Project Setup)."),
})
a = assumptions
st.caption(f"Lifecycle = CAPEX + OPEX over {a.design_life_years:.0f} years + decommissioning "
           f"({a.decommissioning_pct_of_capex:.0f}% of CAPEX). PV discounts future costs at "
           f"{a.discount_rate_pct:.1f}%/yr." + ("" if project.budget else " Set a budget on Project Setup to "
                                                                       "compare against it."))
st.download_button("Download cost summary (CSV)", summary.to_csv(index=False).encode(), "route_costs.csv",
                   "text/csv", icon=":material/download:")

st.plotly_chart(cost_breakdown_chart({names[rid]: e.by_group() for rid, e in estimates.items()}, cur, fx),
                width="stretch")

# --- One route in detail ---------------------------------------------------------
st.divider()
choices = [rid for rid in estimates if rid != "REF"] + ["REF"]
rid = st.selectbox("Route to examine", choices, format_func=names.get)
e = estimates[rid]
route = routes.set_index("route_id").loc[rid]

def millions(value_cad: float) -> str:
    return f"{value_cad * fx / 1e6:,.1f} M"


m1, m2, m3, m4, m5 = st.columns(5)
m1.metric(f"CAPEX ({cur})", millions(e.capex))
m2.metric(f"CAPEX per {unit} ({cur})", millions(e.capex_per_km / to_unit))
m3.metric(f"OPEX per year ({cur})", millions(e.opex_annual))
m4.metric(f"Lifecycle cost PV ({cur})", millions(e.lifecycle_pv))
if project.budget:
    diff = e.capex * fx - project.budget
    # A negative delta (under budget) shows green with a down arrow.
    m5.metric("CAPEX vs budget", f"{100 * e.capex * fx / project.budget:.0f}%",
              f"{diff / 1e6:+,.1f} M {cur}", delta_color="inverse")
else:
    m5.metric("CAPEX vs budget", "no budget set")
for note in e.notes[1:]:
    st.caption(f"• {note}")

lines = e.lines.assign(amount=lambda d: d["amount_cad"] * fx / 1e6,
                       share=lambda d: d["amount_cad"] / e.capex)
st.dataframe(lines[["group", "item", "basis", "amount", "share"]], hide_index=True, width="stretch",
             column_config={"group": "Group", "item": "Item", "basis": st.column_config.TextColumn("Basis (CAD)", width="large"),
                            "amount": st.column_config.NumberColumn(f"Amount ({cur} M)", format="%.2f"),
                            "share": st.column_config.ProgressColumn("Share of CAPEX", min_value=0, max_value=1,
                                                                     format="percent")})
st.download_button("Download this route's cost lines (CSV)", lines.to_csv(index=False).encode(),
                   f"cost_lines_{rid}.csv", "text/csv")

# --- Sensitivity -----------------------------------------------------------------
st.subheader("Sensitivity analysis")
c1, c2 = st.columns([1, 2])
swing = c1.slider("Move each assumption by ±", 5, 50, 25, step=5, format="%d%%")
metric = c2.radio("Result", list(SENSITIVITY_METRICS), format_func=SENSITIVITY_METRICS.get, horizontal=True)
sens = sensitivity(route, project.diameter_mm, assumptions, swing, metric)
st.plotly_chart(tornado_chart(sens, SENSITIVITY_METRICS[metric], cur, fx), width="stretch")
top = sens.iloc[0]
st.caption(f"Base {SENSITIVITY_METRICS[metric].lower()}: {store.fmt_money(sens['base'].iloc[0] * fx, cur)}. "
           f"The most influential assumption is **{top['assumption']}**: ±{swing}% moves the result by about "
           f"{store.fmt_money(top['swing'] / 2 * fx, cur)} either way. Firm up the top assumptions first.")

# --- Lifecycle over time -----------------------------------------------------------
st.subheader("Cost over the design life")
discounted = st.toggle("Show present value (discounted)", value=False)
series = {names[r]: cumulative_cost(estimates[r], assumptions, discounted) for r in estimates if r != "REF"}
st.plotly_chart(cumulative_cost_chart(series, {names[r]: colors[r] for r in estimates}, cur, fx, discounted),
                width="stretch")
st.caption("Year 0 = CAPEX (construction assumed in a single year). Energy costs are excluded unless entered. "
           "No inflation, financing, tax or revenue is modelled.")
