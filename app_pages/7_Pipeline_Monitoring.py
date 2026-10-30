"""FlowGuard Monitoring: simulated readings, explainable alerts, optional ML, acknowledgements."""

import pandas as pd
import streamlit as st

from modules import alert_acks as acks_store
from modules import ui
from modules.anomaly_detection import events_at, section_status
from modules.charts import ml_score_chart, section_detail_chart, status_strip, trend_chart
from modules.config import MONITORING_DISCLAIMER
from modules.detector_evaluation import EVALUATION_NOTE, compare_detectors
from modules.ml_detection import FEATURE_LABELS, ML_DETECTOR, MODELS, ml_events
from modules.monitoring_engine import SCENARIOS, injected_truth

ss = st.session_state

ui.page_header("FlowGuard Monitoring",
               "Simulated pressure, temperature, flow and sensor-health monitoring with explainable alerts.")
ui.synthetic_banner()
st.warning(MONITORING_DISCLAIMER, icon=":material/warning:")

# --- Detection settings ------------------------------------------------------------
with st.expander("Rule thresholds (demonstration values, editable)", icon=":material/tune:"):
    t1, t2, t3 = st.columns(3)
    t1.number_input("High pressure: % above expected", 1.0, 50.0, step=1.0, key="pg_thr_pressure_high_pct")
    t1.number_input("Low pressure: % below expected", 1.0, 50.0, step=1.0, key="pg_thr_pressure_low_pct")
    t1.number_input("Rapid change: % per reading", 0.5, 30.0, step=0.5, key="pg_thr_pressure_rate_pct")
    t2.number_input("Temperature deviation (°C)", 1.0, 50.0, step=1.0, key="pg_thr_temperature_deviation_c")
    t2.number_input("Flow imbalance limit (%)", 0.5, 30.0, step=0.5, key="pg_thr_flow_imbalance_pct")
    t2.number_input("Imbalance must persist (readings)", 1, 20, key="pg_thr_flow_imbalance_min_readings")
    t3.number_input("Minimum sensor health", 0.0, 100.0, step=5.0, key="pg_thr_sensor_health_min")
    t3.number_input("Stuck-sensor window (readings)", 3, 40, key="pg_thr_stuck_window_readings")
    t3.number_input("Merge events separated by ≤ (readings)", 0, 10, key="pg_thr_merge_gap_readings")
    st.caption("Expected pressure for each section = inlet pressure − (pressure loss per km × distance). "
               "Thresholds are illustrative and are not engineering set-points.")

config = ui.get_sim_config()
sections, readings, events = ui.run_monitoring(config, ui.get_thresholds())
timestamps = sorted(readings["timestamp"].unique())

with st.expander("Experimental machine-learning detection (optional)", icon=":material/model_training:"):
    st.toggle("Enable ML anomaly detection", key="pg_ml_enabled",
              help="Adds an unsupervised scikit-learn model alongside the rules. ML findings are advisory only.")
    m1, m2, m3, m4 = st.columns(4)
    m1.selectbox("Model", list(MODELS), format_func=MODELS.get, key="pg_ml_model")
    m2.slider("Baseline (hours learned as normal)", 6, max(6, min(48, config.duration_hours - 6)), key="pg_ml_baseline")
    m3.slider("Threshold margin", 1.0, 2.0, step=0.05, key="pg_ml_margin",
              help="Alert threshold = 99.9th percentile of baseline scores × margin. Higher = fewer alerts.")
    m4.slider("Must persist (readings)", 1, 8, key="pg_ml_min")
    st.caption("Robust covariance measures distance from normal, so slow drifts keep scoring higher; Isolation "
               "Forest is good at odd combinations but its score stops growing outside the range seen in the "
               "baseline. ML findings never raise a section above *Advisory* and never trigger any action.")

ml_enabled = ss["pg_ml_enabled"]
ml_settings = ui.get_ml_settings()
ml_result = None
if ml_enabled:
    if config.scenario != "Normal operations" and config.onset_hour < ml_settings.baseline_hours:
        st.warning(f"The injected event starts at hour {config.onset_hour:g}, inside the ML baseline "
                   f"({ml_settings.baseline_hours:g} h). The model will partly learn the event as 'normal'.",
                   icon=":material/info:")
    ml_result = ui.run_ml_detection(config, ml_settings)
ml_all = ml_events(ml_result, ml_settings) if ml_result is not None else None

cfg_key = acks_store.config_key(config)
section_ids = sections["section_id"].tolist()
unit = config.profile.flow_unit

# --- View controls ---------------------------------------------------------------------
default_section = f"S{config.affected_section:02d}"
if ss.get("pg_selected_section") not in section_ids:
    ss["pg_selected_section"] = default_section
if ss.get("mon_idx", 10**9) >= len(timestamps):
    ss["mon_idx"] = len(timestamps) - 1

c1, c2, c3 = st.columns([1, 1, 3])
c1.selectbox("Section to inspect", section_ids, key="pg_selected_section")
live = c2.toggle("Live-style replay", key="mon_live",
                 help="Steps through the simulated window automatically, one reading (15 min) per second.")
if live:
    if "mon_live_idx" not in ss:
        ss["mon_live_idx"] = max(0, int(config.onset_hour * 60 / config.interval_minutes) - 16)
    c3.caption("Replay running. Turn the toggle off to pick a time manually.")
else:
    ss.pop("mon_live_idx", None)
    c3.slider("Replay time (drag to review any moment in the window)", 0, len(timestamps) - 1, key="mon_idx",
              format="reading %d")


def _alert_card(ev: pd.Series) -> None:
    is_ml = ev["detector"] == ML_DETECTOR
    icon = ":material/psychology:" if is_ml else (":material/error:" if ev["priority"] >= 3 else ":material/warning:")
    with st.container(border=True):
        top, act = st.columns([5, 1.3], vertical_alignment="center")
        top.markdown(f"{icon} **{ev['severity']}** · **{ev['section_id']}** · {ev['rule']} · *{ev['detector']}* "
                     f"&nbsp;<span style='opacity:0.7'>(since {ev['start']:%b %d %H:%M}) · {ev['ack_status']}</span>",
                     unsafe_allow_html=True)
        top.caption(f"Why: {ev['explanation']}" + (f"  \nOperator note: {ev['ack_note']}" if ev["ack_note"] else ""))
        with act.popover("Acknowledge", icon=":material/how_to_reg:", width="stretch"):
            with st.form(f"ack_{ev['ack_key']}"):
                status = st.selectbox("Status", acks_store.ACK_STATUSES,
                                      index=acks_store.ACK_STATUSES.index(ev["ack_status"]))
                note = st.text_area("Note", value=ev["ack_note"] or "")
                actor = st.text_input("Your name or role (optional)")
                st.caption("Record-keeping only: acknowledging never silences a rule or triggers an action.")
                if st.form_submit_button("Save", type="primary"):
                    acks_store.set_ack(ev["ack_key"], status, note, actor)
                    st.rerun()  # full rerun closes the popover and refreshes the event history


@st.fragment(run_every=1.0 if live else None)
def live_panel() -> None:
    if live:
        ss["mon_live_idx"] = (ss.get("mon_live_idx", 0) + 1) % len(timestamps)
        idx = ss["mon_live_idx"]
    else:
        idx = ss["mon_idx"]
    now = pd.Timestamp(timestamps[idx])
    sid = ss["pg_selected_section"]

    # Only what was knowable at `now` (no look-ahead).
    current = events_at(ui.rule_events_as_of(config, ui.get_thresholds(), now), now)
    if ml_result is not None:
        current = pd.concat([current, events_at(ml_events(ml_result, ml_settings, as_of=now), now)],
                            ignore_index=True)
    statuses = section_status(sections, current)

    st.markdown(f"#### Status at {now:%Y-%m-%d %H:%M} &nbsp;·&nbsp; hour {idx * config.interval_minutes / 60:.2f} "
                f"of {config.duration_hours}")
    st.plotly_chart(status_strip(sections, statuses), width="stretch", key="strip")

    sec = readings[readings["section_id"] == sid].set_index("timestamp")
    row = sec.loc[now]
    prev = sec.loc[pd.Timestamp(timestamps[max(idx - 1, 0)])]
    status = statuses.set_index("section_id").loc[sid, "status"]

    def delta(col, fmt):
        d = row[col] - prev[col]
        return None if pd.isna(d) else fmt.format(d)

    st.markdown(f"**Section {sid}** &nbsp;{ui.status_badge(status)}", unsafe_allow_html=True)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Pressure", "no data" if pd.isna(row["pressure_kpa"]) else f"{row['pressure_kpa']:,.0f} kPa",
              delta("pressure_kpa", "{:+,.0f} kPa"), delta_color="off")
    m2.metric("Temperature", "no data" if pd.isna(row["temperature_c"]) else f"{row['temperature_c']:.1f} °C",
              delta("temperature_c", "{:+.1f} °C"), delta_color="off")
    m3.metric("Inlet flow", f"{row['flow_in']:,.0f} {unit}", delta("flow_in", "{:+,.0f}"), delta_color="off")
    m4.metric("Flow imbalance", f"{row['flow_imbalance_pct']:.2f} %", delta("flow_imbalance_pct", "{:+.2f} pts"),
              delta_color="inverse")
    m5.metric("Sensor health", f"{row['sensor_health']:.0f}/100", delta("sensor_health", "{:+.1f}"))

    st.markdown("#### Active alerts")
    st.caption("Unacknowledged first, then by priority. Rules drive status; ML findings are advisory.")
    if current.empty:
        st.success("No rule (or ML flag) is active on any section.", icon=":material/check_circle:")
    else:
        shown = acks_store.with_acks(current, cfg_key, acks_store.acks_for(cfg_key))
        shown["unacked"] = shown["ack_status"] == "New"
        for _, ev in shown.sort_values(["unacked", "priority", "start"], ascending=[False, False, True]).iterrows():
            _alert_card(ev)

    tab_names = ["Trends: all sections", f"Section {sid} detail"] + (["ML anomaly score"] if ml_result else [])
    tabs = st.tabs(tab_names)
    with tabs[0]:
        a, b = st.columns(2)
        a.plotly_chart(trend_chart(readings, "pressure_kpa", "Pressure", "kPa", sid, now), width="stretch", key="tp")
        b.plotly_chart(trend_chart(readings, "flow_imbalance_pct", "Flow imbalance (inlet − outlet)", "%", sid, now),
                       width="stretch", key="tf")
        a.plotly_chart(trend_chart(readings, "temperature_c", "Temperature", "°C", sid, now), width="stretch", key="tt")
        b.plotly_chart(trend_chart(readings, "sensor_health", "Sensor health", "score", sid, now),
                       width="stretch", key="th")
    with tabs[1]:
        thr = ui.get_thresholds()
        exp_p = float(sections.set_index("section_id").loc[sid, "expected_pressure_kpa"])
        st.plotly_chart(
            section_detail_chart(
                sec.reset_index(), events[events["section_id"] == sid], exp_p, thr.pressure_high_pct,
                thr.pressure_low_pct, float(sections.set_index("section_id").loc[sid, "expected_temperature_c"]),
                thr.temperature_deviation_c, unit, now,
            ),
            width="stretch", key="detail",
        )
        st.caption("Dashed red lines = rule thresholds. Shaded bands = rule events over the whole window "
                   "(colour = severity).")
    if ml_result is not None:
        with tabs[2]:
            sc = ml_result.scores[ml_result.scores["section_id"] == sid]
            st.plotly_chart(ml_score_chart(sc, ml_result.threshold, ml_result.baseline_end, sid, now),
                            width="stretch", key="ml")


live_panel()

# --- History, comparison and model notes ------------------------------------------------
st.divider()
all_events = events if ml_all is None else pd.concat([events, ml_all], ignore_index=True)
all_events = acks_store.with_acks(all_events, cfg_key, acks_store.acks_for(cfg_key))
h_tab, c_tab, ml_tab = st.tabs(["Event history", "Detector comparison (simulated ground truth)", "About the ML model"])

with h_tab:
    if all_events.empty:
        st.info("No events were detected in this window.")
    else:
        f1, f2, f3, f4 = st.columns(4)
        sev_filter = f1.multiselect("Severity", ["Critical", "High", "Medium", "Low"],
                                    default=["Critical", "High", "Medium", "Low"])
        sec_filter = f2.multiselect("Sections", section_ids, default=sorted(all_events["section_id"].unique()))
        det_filter = f3.multiselect("Detector", sorted(all_events["detector"].unique()),
                                    default=sorted(all_events["detector"].unique()))
        ack_filter = f4.multiselect("Acknowledgement", acks_store.ACK_STATUSES, default=acks_store.ACK_STATUSES)
        shown = all_events[all_events["severity"].isin(sev_filter) & all_events["section_id"].isin(sec_filter)
                           & all_events["detector"].isin(det_filter) & all_events["ack_status"].isin(ack_filter)]
        st.dataframe(
            shown.drop(columns=["priority", "category", "ack_key"]),
            hide_index=True, width="stretch",
            column_config={
                "start": st.column_config.DatetimeColumn("Start", format="MMM D HH:mm"),
                "end": st.column_config.DatetimeColumn("End", format="MMM D HH:mm"),
                "duration_min": st.column_config.NumberColumn("Duration (min)", format="%.0f"),
                "peak_value": st.column_config.NumberColumn("Peak", format="%.2f"),
                "active": st.column_config.CheckboxColumn("Still active"),
                "explanation": st.column_config.TextColumn("Why it was raised", width="large"),
                "ack_status": "Acknowledgement", "ack_note": "Operator note",
            },
        )
        st.download_button("Download event history (CSV)", shown.drop(columns=["ack_key"]).to_csv(index=False).encode(),
                           "flowguard_events_SYNTHETIC.csv", "text/csv", icon=":material/download:")

with c_tab:
    truth = injected_truth(config)
    st.caption(EVALUATION_NOTE)
    if truth.empty:
        st.markdown("No event was injected (normal operations), so every alert counts as a false alarm.")
    else:
        st.markdown(f"**Injected:** {config.scenario} ({truth['kind'].iloc[0].lower()}) on "
                    f"{', '.join(truth['section_id'])} from {truth['start'].iloc[0]:%b %d %H:%M}. "
                    f"{SCENARIOS[config.scenario]}")
    comparison = compare_detectors(events, ml_all, truth)
    comparison["detected"] = comparison["detected"].map({True: "Yes", False: "No", None: "n/a"})
    comparison["delay_hours"] = comparison["delay_hours"].map(lambda h: "—" if h is None or pd.isna(h) else f"{h:.2f}")
    st.dataframe(comparison, hide_index=True, width="stretch", column_config={
        "detector": "Detector", "alerts": "Alerts raised", "false_alarms": "Alerts not matching the injected event",
        "detected": "Detected the injected event?",
        "delay_hours": "Hours from start to first matching alert"})
    if not ml_enabled:
        st.caption("Enable ML detection above to compare it with the rules.")
    st.caption("Try *Flow meter fault (bias)* (rules flag a 'possible leak'; the cross-check points to the meter) "
               "and *Pressure sensor drift* (the rules miss it; the robust-covariance ML usually catches it).")

with ml_tab:
    if ml_result is None:
        st.info("ML detection is off. Enable it in *Experimental machine-learning detection* above.")
    else:
        for note in ml_result.notes:
            st.markdown(f"- {note}")
        st.markdown(f"- Alert threshold: anomaly score > **{ml_result.threshold:.3f}** "
                    f"(99.9th percentile of baseline scores × {ml_settings.threshold_margin}).")
        st.markdown(f"- A flag must persist for {ml_settings.min_consecutive} readings to become an event.")
        st.markdown("- Inputs: " + ", ".join(FEATURE_LABELS.values()) + ".")
        st.markdown("- Explanations list the inputs that departed most from the baseline (in standard deviations). "
                    "They are approximate: the model does not give exact reasons.")
        st.markdown("- Limits: learns only what the baseline contained; an event inside the baseline is learned as "
                    "normal; operating changes (new flow rates, seasons) would look anomalous; it is not validated "
                    "and cannot detect real leaks.")

with st.expander("Raw simulated readings"):
    st.dataframe(readings, hide_index=True, width="stretch", height=300)
    st.download_button("Download readings (CSV)", readings.to_csv(index=False).encode(),
                       "flowguard_readings_SYNTHETIC.csv", "text/csv")
