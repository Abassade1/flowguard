"""Indigenous & Community Engagement (Module E): records of engagement, concerns, commitments and changes."""

from datetime import date

import pandas as pd
import streamlit as st

from modules import social_impact as si
from modules import ui
from modules.routing import STRATEGIES

ss = st.session_state
project = ui.get_active_project()
ROUTE_OPTIONS = {s.key: s.name for s in STRATEGIES}

ui.page_header("Indigenous & Community Engagement",
               "Keep a clear record of engagement, what was heard, what was promised, and what changed.")

with st.container(border=True):
    st.markdown("**How this module works**")
    c1, c2 = st.columns(2)
    c1.markdown(
        "- Indigenous rights and community authority come first. This tool only **keeps records**; it never "
        "scores, weighs or puts a price on rights, cultural values or community positions.\n"
        "- **Engagement records are not consent.** Nothing here measures consent, support or agreement.\n"
        "- Decisions belong to people and to the communities' own processes, not to this software.")
    c2.markdown(
        "- **No locations** are stored for land use or traditional activities, and the app never infers "
        "territories or cultural sites.\n"
        "- Record only **authorised** information. Communities' data belongs to them (data sovereignty, e.g. "
        "OCAP® principles for First Nations data).\n"
        "- *Restricted* records are hidden on screen by default and **never exported**. This local app has no "
        "user accounts: these are handling rules, not security controls.")

if project.id is None:
    st.warning("Save the project on Project Setup first. Engagement records are stored with a saved project.",
               icon=":material/save:")
    st.page_link("app_pages/1_Project_Setup.py", label="Go to Project Setup", icon=":material/arrow_forward:")
    st.stop()

records = ui.engagement_records(project)
show_restricted = st.toggle("Show restricted content on this screen", value=False,
                            help="Reveals Restricted records for this session only. They are still excluded "
                                 "from every export.")
view = si.redact(records, show_restricted)

s = si.summarise(records)
m = st.columns(6)
m[0].metric("Activities", s["activities"])
m[1].metric("Open concerns", s["open_concerns"], help=f"Concerns not marked Resolved, of {s['concerns']} raised. "
            "Only those who raised a concern can say whether it is resolved.")
m[2].metric("Disagreements", s["unresolved"], help="Concerns recorded as 'Unresolved (disagreement remains)'.")
m[3].metric("Open items", s["open_items"], help="Commitments and follow-up actions not completed or cancelled.")
m[4].metric("Overdue", s["overdue"], help="Open commitments and actions past their due date.")
m[5].metric("Changes", s["changes"], help="Project changes documented as arising from engagement.")


def _record_label(row) -> str:
    return f"#{row['id']} {si.RECORD_TYPES[row['record_type']].singular}: {row['title']}"


def _input(form_key: str, col: str, label: str, rtype: si.RecordType, current=None):
    """Draw the right input for a column; returns the value to store."""
    key = f"{form_key}_{col}"
    if col in ("record_date", "due_date"):
        value = pd.to_datetime(current).date() if current else (date.today() if col == "record_date" else None)
        picked = st.date_input(label, value=value, key=key, format="YYYY-MM-DD")
        return picked.isoformat() if picked else None
    if col == "category":
        options = list(rtype.categories)
        return st.selectbox(label, options, index=options.index(current) if current in options else 0, key=key)
    if col == "status":
        options = list(rtype.statuses)
        return st.selectbox(label, options, index=options.index(current) if current in options else 0, key=key)
    if col == "linked_route":
        options = [None, *ROUTE_OPTIONS]
        picked = st.selectbox(label, options, index=options.index(current) if current in options else 0,
                              format_func=lambda k: "(none)" if k is None else ROUTE_OPTIONS[k], key=key)
        return picked
    if col == "linked_record_id":
        candidates = view[view["record_type"].isin(rtype.link_types)]
        labels = {int(r["id"]): _record_label(r) for _, r in candidates.iterrows()}
        options = [None, *labels]
        current = int(current) if current is not None and not pd.isna(current) else None
        return st.selectbox(label, options, index=options.index(current) if current in options else 0,
                            format_func=lambda k: "(none)" if k is None else labels[k], key=key)
    if col in ("details", "response", "authorization"):
        return st.text_area(label, value=current or "", key=key, height=90)
    return st.text_input(label, value=current or "", key=key)


tabs = st.tabs([rt.label for rt in si.RECORD_TYPES.values()] + ["Export & data governance"])
for rtype, tab in zip(si.RECORD_TYPES.values(), tabs):
    with tab:
        if rtype.guidance:
            st.caption(rtype.guidance)
        mine = view[view["record_type"] == rtype.key].copy()
        if mine.empty:
            st.info(f"No {rtype.label.lower()} recorded yet.")
        else:
            if rtype.key in ("commitment", "action"):
                mine["overdue"] = si.is_overdue(mine)
            if "linked_route" in rtype.fields:
                mine["linked_route"] = mine["linked_route"].map(lambda k: ROUTE_OPTIONS.get(k, ""))
            cols = ["id", *rtype.fields, "sensitivity"] + (["overdue"] if "overdue" in mine else [])
            st.dataframe(mine[cols].rename(columns={**rtype.fields, "id": "#", "sensitivity": "Sensitivity",
                                                    "overdue": "Overdue"}),
                         hide_index=True, width="stretch")

        with st.expander(f"Add {rtype.singular.lower()}", icon=":material/add:"):
            with st.form(f"add_{rtype.key}", clear_on_submit=True):
                values = {col: _input(f"add_{rtype.key}", col, label, rtype) for col, label in rtype.fields.items()}
                values["sensitivity"] = st.selectbox(
                    "Sensitivity", si.SENSITIVITY_LEVELS, index=si.SENSITIVITY_LEVELS.index(rtype.default_sensitivity),
                    key=f"add_{rtype.key}_sensitivity",
                    help="Restricted: hidden by default and never exported.")
                if st.form_submit_button(f"Add {rtype.singular.lower()}", type="primary"):
                    try:
                        si.add_record(project.id, rtype.key, values)
                        st.toast(f"{rtype.singular} added", icon=":material/check_circle:")
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))

        if not mine.empty:
            with st.expander(f"Update or delete a {rtype.singular.lower()}", icon=":material/edit:"):
                raw = records[records["record_type"] == rtype.key].set_index("id")
                rid = st.selectbox("Record", list(raw.index), key=f"pick_{rtype.key}",
                                   format_func=lambda i: _record_label(view.set_index("id").loc[i].to_dict() | {"id": i}))
                rec = raw.loc[rid]
                hidden = rec["sensitivity"] == si.RESTRICTED and not show_restricted
                if hidden:
                    st.info("Turn on *Show restricted content* to edit this record.")
                else:
                    form_key = f"upd_{rtype.key}_{rid}"
                    with st.form(form_key):
                        changes = {col: _input(form_key, col, rtype.fields[col], rtype, rec[col])
                                   for col in rtype.fields if col in rtype.updatable}
                        changes["sensitivity"] = st.selectbox("Sensitivity", si.SENSITIVITY_LEVELS,
                                                              index=si.SENSITIVITY_LEVELS.index(rec["sensitivity"]),
                                                              key=f"{form_key}_sensitivity")
                        if st.form_submit_button("Save changes", type="primary"):
                            si.update_record(int(rid), changes)
                            st.toast("Record updated", icon=":material/check_circle:")
                            st.rerun()
                    confirm = st.checkbox("I want to permanently delete this record", key=f"confirm_{form_key}")
                    if st.button("Delete record", disabled=not confirm, key=f"del_{form_key}"):
                        si.delete_record(int(rid))
                        st.rerun()

with tabs[-1]:
    exportable = si.export_view(records)
    withheld = len(records) - len(exportable)
    st.markdown(
        "Exports contain Public and Internal records only. Before sharing any export outside the project team, "
        "check it against the terms under which information was shared and any data-sharing agreement with "
        "the communities concerned.")
    st.metric("Records that will be exported", len(exportable), f"{withheld} restricted record(s) withheld",
              delta_color="off", delta_arrow="off")
    st.download_button("Download engagement records (CSV, restricted excluded)",
                       exportable.to_csv(index=False).encode(), "engagement_records.csv", "text/csv",
                       icon=":material/download:", disabled=exportable.empty)
