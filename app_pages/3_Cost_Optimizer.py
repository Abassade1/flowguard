"""Cost Optimizer (Module C): editable assumptions, CAPEX/OPEX/lifecycle, breakdown and sensitivity."""

from dataclasses import fields

import pandas as pd
import streamlit as st

from modules import projects as store
from modules import ui
from modules import cost_uncertainty as mc
from modules.charts import cost_breakdown_chart, cumulative_cost_chart, tornado_chart
from modules.charts import s_curve_chart as mc_chart
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

# --- Probabilistic estimate (Monte Carlo) -----------------------------------------------
st.subheader("Probabilistic estimate (Monte Carlo)")
st.caption("Instead of one 'most likely' number plus a flat contingency, each uncertain input gets a range and the "
           "estimate is recalculated thousands of times. P80 means 80% of simulated outcomes cost that much or less. "
           "Contingency is not simulated: the results show how much contingency each confidence level needs.")

ranges = mc.ranges_with_overrides(project.cost_uncertainty)
labels = mc.range_labels()
with st.expander("Uncertainty ranges (editable)", icon=":material/tune:"):
    st.caption("Low and high are % changes from the most likely value (triangular distribution). Set both to 0 to "
               "treat an input as certain. Saved with the project.")
    table = pd.DataFrame([{"key": k, "input": labels[k],
                           "type": "Quantity" if k in mc.QUANTITY_LABELS else "Assumption",
                           "low_pct": lo, "high_pct": hi} for k, (lo, hi) in ranges.items()])
    edited = st.data_editor(
        table, hide_index=True, width="stretch", disabled=["key", "input", "type"],
        column_order=["input", "type", "low_pct", "high_pct"], key=f"mc_ranges_{ss['pg_project_rev']}",
        column_config={"input": "Input", "type": "Type",
                       "low_pct": st.column_config.NumberColumn("Low (%)", min_value=-90.0, max_value=0.0, step=5.0),
                       "high_pct": st.column_config.NumberColumn("High (%)", min_value=0.0, max_value=300.0, step=5.0)})
    defaults = mc.default_ranges()
    overrides = {r["key"]: [float(r["low_pct"]), float(r["high_pct"])] for _, r in edited.iterrows()
                 if (float(r["low_pct"]), float(r["high_pct"])) != tuple(map(float, defaults[r["key"]]))}
    if overrides != project.cost_uncertainty:
        project.cost_uncertainty = overrides
        ui.update_active_project(project)
        st.rerun()
    c1, c2 = st.columns(2)
    runs = c1.select_slider("Number of simulations", [1000, 2000, 5000, 10000, 20000], value=5000)
    seed = c2.number_input("Random seed", 0, 9999, 1, help="Same seed and inputs give identical results.")

results = {rid_: mc.simulate(routes.set_index("route_id").loc[rid_], project.diameter_mm, assumptions, ranges,
                             runs, seed)
           for rid_ in estimates if rid_ != "REF"}
budget_cad = project.budget / fx if project.budget else None
st.plotly_chart(mc_chart({names[r]: res.samples["capex_base"].to_numpy() for r, res in results.items()},
                         {names[r]: colors[r] for r in results}, cur, fx, project.budget or None), width="stretch")

summary_mc = pd.DataFrame([{
    "Route": names[r],
    **{f"{row['level']} ({cur} M)": row["capex_base"] * fx / 1e6 for _, row in res.percentiles.iterrows()},
    "Chance within budget": res.probability_within(budget_cad) if budget_cad else None,
    f"Your {assumptions.contingency_pct:.0f}% contingency covers": res.confidence_of_assumed_contingency,
} for r, res in results.items()])
st.dataframe(summary_mc, hide_index=True, width="stretch", column_config={
    **{c: st.column_config.NumberColumn(c, format="%.0f") for c in summary_mc.columns if "(" in c},
    "Chance within budget": st.column_config.ProgressColumn("Chance within budget", min_value=0, max_value=1,
                                                            format="percent"),
    f"Your {assumptions.contingency_pct:.0f}% contingency covers": st.column_config.ProgressColumn(
        f"Your {assumptions.contingency_pct:.0f}% contingency covers", min_value=0, max_value=1, format="percent",
        help="Share of simulated outcomes at or below most-likely cost + your contingency."),
})

if rid in results:
    res = results[rid]
    p = res.percentiles.set_index("level")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"P50 CAPEX ({cur})", millions(p.loc["P50", "capex_base"]))
    m2.metric(f"P80 CAPEX ({cur})", millions(p.loc["P80", "capex_base"]))
    m3.metric("Contingency needed for P80", f"{p.loc['P80', 'implied_contingency_pct']:.0f}%",
              f"assumed {assumptions.contingency_pct:.0f}%", delta_color="off", delta_arrow="off")
    m4.metric(f"Lifecycle PV at P80 ({cur})", millions(p.loc["P80", "lifecycle_pv_base"]))
    coverage = res.confidence_of_assumed_contingency
    st.info(f"For {names[rid]}: the most-likely estimate before contingency is {millions(res.most_likely_base)} {cur}. "
            f"Your {assumptions.contingency_pct:.0f}% contingency covers about {coverage:.0%} of simulated outcomes "
            f"(roughly P{coverage * 100:.0f}); reaching P80 needs about {p.loc['P80', 'implied_contingency_pct']:.0f}%.",
            icon=":material/insights:")
st.caption("Limits: inputs are drawn independently, which tends to understate the spread (in practice, labour and "
           "material prices often move together); ranges are illustrative defaults; scope changes, schedule delays "
           "and regulatory outcomes are not included. Lifecycle values here exclude contingency.")

# --- Lifecycle over time -----------------------------------------------------------
st.subheader("Cost over the design life")
discounted = st.toggle("Show present value (discounted)", value=False)
series = {names[r]: cumulative_cost(estimates[r], assumptions, discounted) for r in estimates if r != "REF"}
st.plotly_chart(cumulative_cost_chart(series, {names[r]: colors[r] for r in estimates}, cur, fx, discounted),
                width="stretch")
st.caption("Year 0 = CAPEX (construction assumed in a single year). Energy costs are excluded unless entered. "
           "No inflation, financing, tax or revenue is modelled.")
