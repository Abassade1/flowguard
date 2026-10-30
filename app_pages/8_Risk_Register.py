"""Risk & Mitigation Register: people rate risks; the app suggests what it can see in the project data."""

import pandas as pd
import streamlit as st

from modules import risk_register as rr
from modules import ui
from modules.charts import risk_matrix_chart
from modules.routing import STRATEGIES

ss = st.session_state
project = ui.get_active_project()
ROUTE_OPTIONS = {s.key: s.name for s in STRATEGIES}

ui.page_header("Risk & Mitigation Register",
               "Engineering, environmental, social, cost, regulatory, operational and data risks in one place.")
st.caption("Ratings are professional judgement on a 5 × 5 likelihood × consequence scale, before (inherent) and "
           "after (residual) mitigation. The app suggests risks it can see in the project data; nothing enters the "
           "register until a person reviews and rates it. Social and Indigenous matters are obligations to address "
           "through engagement, never costs to be offset.")
ui.require_valid_project(project)
if project.id is None:
    st.warning("Save the project on Project Setup first. Risks are stored with a saved project.", icon=":material/save:")
    st.page_link("app_pages/1_Project_Setup.py", label="Go to Project Setup", icon=":material/arrow_forward:")
    st.stop()

a = ui.project_analysis(project)
risks = a.risks
open_risks = risks[risks["status"] != "Closed"]

m = st.columns(4)
m[0].metric("Open risks", len(open_risks))
m[1].metric("High or very high (residual)", int(open_risks["residual_level"].isin(["High", "Very high"]).sum()))
m[2].metric("Mitigations not started", int((open_risks["mitigation_status"] == "Not started").sum()))
m[3].metric("Closed", int((risks["status"] == "Closed").sum()))

c1, c2 = st.columns(2)
c1.plotly_chart(risk_matrix_chart(rr.matrix_counts(risks), "Inherent rating (before mitigation)"), width="stretch")
c2.plotly_chart(risk_matrix_chart(rr.matrix_counts(risks, residual=True), "Residual rating (after mitigation)"),
                width="stretch")
st.caption("Numbers are counts of open risks. Colours: green Low (1-4), amber Medium (5-9), red High (10-14), "
           "dark red Very high (15-25).")


def _rating_inputs(prefix: str, defaults: dict | None = None) -> dict:
    d = defaults or {}
    c = st.columns(4)
    pick = {}
    for col, (key, label, names) in zip(c, [("likelihood", "Likelihood", rr.LIKELIHOOD_LABELS),
                                            ("consequence", "Consequence", rr.CONSEQUENCE_LABELS),
                                            ("residual_likelihood", "Residual likelihood", rr.LIKELIHOOD_LABELS),
                                            ("residual_consequence", "Residual consequence", rr.CONSEQUENCE_LABELS)]):
        current = d.get(key)
        current = int(current) if current is not None and not pd.isna(current) else 3
        pick[key] = col.selectbox(label, list(names), index=current - 1, format_func=lambda v, n=names: f"{v} {n[v]}",
                                  key=f"{prefix}_{key}")
    return pick


# --- Suggestions ----------------------------------------------------------------------
taken = rr.accepted_keys(risks) | set(project.dismissed_risk_suggestions)
suggestions = [s for s in rr.suggest_risks(
    a.routes, a.screening, a.design["strategy"], a.engagement, project.budget / max(ui.store.to_currency(1.0, project), 1e-9),
    dict(zip(a.routes["strategy"], a.routes["capex_cad"])),
    a.sensitivity["assumption"].iloc[0] if not a.sensitivity.empty else None, a.plan.register) if s.key not in taken]

with st.expander(f"Suggested risks to review ({len(suggestions)})", icon=":material/lightbulb:",
                 expanded=bool(suggestions) and risks.empty):
    st.caption(f"Based on the modelled route ({a.design['name']}) and project data. Accept to add with your ratings, "
               "or set aside if not relevant (remembered when you save the project).")
    if not suggestions:
        st.info("No new suggestions.")
    for s in suggestions:
        with st.container(border=True):
            st.markdown(f"**{s.title}** · {s.category} · *from {s.source}*")
            st.caption(s.description)
            with st.form(f"sugg_{s.key}"):
                ratings = _rating_inputs(f"sugg_{s.key}")
                c1, c2 = st.columns(2)
                owner = c1.text_input("Owner", key=f"sugg_{s.key}_owner")
                mitigation = c2.text_input("Planned mitigation", key=f"sugg_{s.key}_mit")
                b1, b2 = st.columns([1, 4])
                if b1.form_submit_button("Add to register", type="primary"):
                    rr.add_risk(project.id, {"title": s.title, "description": s.description, "category": s.category,
                                             "linked_route": s.linked_route, "owner": owner, "mitigation": mitigation,
                                             "source": rr.suggestion_source(s), **ratings})
                    st.toast("Risk added", icon=":material/check_circle:")
                    st.rerun()
                if b2.form_submit_button("Set aside"):
                    project.dismissed_risk_suggestions = [*project.dismissed_risk_suggestions, s.key]
                    ui.update_active_project(project)
                    st.rerun()

# --- Register -------------------------------------------------------------------------
st.subheader("Register")
if risks.empty:
    st.info("No risks recorded yet. Review the suggestions above or add your own below.")
else:
    f1, f2 = st.columns(2)
    cat_filter = f1.multiselect("Category", rr.CATEGORIES, default=rr.CATEGORIES)
    status_filter = f2.multiselect("Status", rr.RISK_STATUSES, default=["Open", "Monitoring"])
    shown = risks[risks["category"].isin(cat_filter) & risks["status"].isin(status_filter)].copy()
    shown["linked_route"] = shown["linked_route"].map(lambda k: ROUTE_OPTIONS.get(k, ""))
    shown = shown.sort_values(["residual_score", "score"], ascending=False)
    st.dataframe(shown[["id", "title", "category", "linked_route", "score", "level", "residual_score", "residual_level",
                        "owner", "mitigation", "mitigation_status", "status", "source"]],
                 hide_index=True, width="stretch",
                 column_config={"id": "#", "score": "Inherent", "level": "Inherent level",
                                "residual_score": "Residual", "residual_level": "Residual level",
                                "linked_route": "Route option", "mitigation_status": "Mitigation status"})
    st.download_button("Download register (CSV)", risks.drop(columns=["project_id"]).to_csv(index=False).encode(),
                       "risk_register.csv", "text/csv", icon=":material/download:")

with st.expander("Add a risk", icon=":material/add:"):
    with st.form("add_risk", clear_on_submit=True):
        title = st.text_input("Risk")
        description = st.text_area("Description (cause, event, effect)", height=80)
        c1, c2 = st.columns(2)
        category = c1.selectbox("Category", rr.CATEGORIES)
        route = c2.selectbox("Related route option", [None, *ROUTE_OPTIONS],
                             format_func=lambda k: "(none)" if k is None else ROUTE_OPTIONS[k])
        ratings = _rating_inputs("new")
        c1, c2 = st.columns(2)
        owner = c1.text_input("Owner")
        mitigation = c2.text_input("Planned mitigation")
        if st.form_submit_button("Add risk", type="primary"):
            try:
                rr.add_risk(project.id, {"title": title, "description": description, "category": category,
                                         "linked_route": route, "owner": owner, "mitigation": mitigation, **ratings})
                st.toast("Risk added", icon=":material/check_circle:")
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))

if not risks.empty:
    with st.expander("Update or delete a risk", icon=":material/edit:"):
        rid = st.selectbox("Risk", list(risks["id"]),
                           format_func=lambda i: f"#{i} {risks.set_index('id').loc[i, 'title']}")
        r = risks.set_index("id").loc[rid].to_dict()
        with st.form(f"edit_risk_{rid}"):
            ratings = _rating_inputs(f"edit_{rid}", r)
            c1, c2, c3 = st.columns(3)
            owner = c1.text_input("Owner", value=r["owner"] or "")
            mit_status = c2.selectbox("Mitigation status", rr.MITIGATION_STATUSES,
                                      index=rr.MITIGATION_STATUSES.index(r["mitigation_status"] or "Not started"))
            status = c3.selectbox("Status", rr.RISK_STATUSES, index=rr.RISK_STATUSES.index(r["status"]))
            mitigation = st.text_area("Mitigation", value=r["mitigation"] or "", height=70)
            if st.form_submit_button("Save changes", type="primary"):
                rr.update_risk(int(rid), {"owner": owner, "mitigation_status": mit_status, "status": status,
                                          "mitigation": mitigation, **ratings})
                st.rerun()
        confirm = st.checkbox("I want to permanently delete this risk", key=f"del_confirm_{rid}")
        if st.button("Delete risk", disabled=not confirm):
            rr.delete_risk(int(rid))
            st.rerun()

# --- Mitigation commitments -------------------------------------------------------------
st.subheader("Mitigation commitments")
from_risks = open_risks.loc[open_risks["mitigation"].fillna("") != "", ["mitigation", "owner", "mitigation_status"]] \
    .rename(columns={"mitigation": "commitment", "mitigation_status": "status"}).assign(origin="Risk register")
from_engagement = a.commitments[a.commitments["category"] == "Mitigation commitment"][["title", "owner", "status"]] \
    .rename(columns={"title": "commitment"}).assign(origin="Engagement commitment")
commitments = pd.concat([from_risks, from_engagement], ignore_index=True)
if commitments.empty:
    st.info("No mitigation commitments yet. Add mitigations to risks, or record mitigation commitments on the "
            "Engagement page.")
else:
    st.dataframe(commitments, hide_index=True, width="stretch")
st.caption("Engagement commitments marked Restricted are not shown here.")
