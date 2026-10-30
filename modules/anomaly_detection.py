"""Transparent, rule-based detection of unusual readings.

Each rule is simple enough to explain in one sentence, and every event records
WHY it was raised (observed value, threshold, duration). Consecutive breaches are
grouped into a single event so the operator sees one alert per incident rather
than one per reading.

This is a demonstration on simulated data. It cannot reliably detect real leaks.
A machine-learning detector (Scikit-learn) is planned for Phase 6 as an optional,
additional layer, not a replacement for these explainable rules.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from modules.config import SEVERITY_PRIORITY

EVENT_COLUMNS = [
    "event_id", "section_id", "rule", "category", "parameter", "severity", "priority",
    "start", "end", "duration_min", "peak_value", "threshold", "active", "explanation", "detector",
]
RULES_DETECTOR = "Rules"

# Rules about the instrument itself rather than the process.
SENSOR_RULES = {"Sensor health low", "Sensor possibly stuck", "Missing readings"}


@dataclass(frozen=True)
class Thresholds:
    """Demonstration thresholds. All are editable from the Monitoring page."""

    pressure_high_pct: float = 10.0  # % above the section's expected pressure
    pressure_low_pct: float = 8.0  # % below the section's expected pressure
    pressure_rate_pct: float = 5.0  # % change between consecutive readings
    temperature_deviation_c: float = 10.0  # °C away from the nominal temperature
    flow_imbalance_pct: float = 3.0  # (inlet - outlet) / inlet
    flow_imbalance_min_readings: int = 3  # must persist this many readings
    sensor_health_min: float = 70.0  # self-reported health score (0-100)
    stuck_window_readings: int = 8  # identical readings in a row => possibly stuck
    merge_gap_readings: int = 2  # breaches separated by <= this many readings merge


def _runs(mask: np.ndarray, merge_gap: int) -> list[tuple[int, int]]:
    """Find (start, end) index pairs of True runs, merging runs separated by small gaps."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    runs = []
    start = prev = idx[0]
    for i in idx[1:]:
        if i - prev > merge_gap + 1:
            runs.append((start, prev))
            start = i
        prev = i
    runs.append((start, prev))
    return runs


def _escalate(base: str, excess_ratio: float) -> str:
    """Raise severity one level when a value is far past its threshold."""
    levels = ["Low", "Medium", "High", "Critical"]
    i = levels.index(base)
    if excess_ratio >= 2.0:
        i = min(i + 1, len(levels) - 1)
    return levels[i]


def _expected_temperature(sections: pd.DataFrame, nominal_temperature_c: float) -> pd.Series:
    """Per-section expected temperature; the inlet (nominal) value if sections don't provide one."""
    s = sections.set_index("section_id")
    if "expected_temperature_c" in s:
        return s["expected_temperature_c"]
    return pd.Series(nominal_temperature_c, index=s.index)


def detect_events(
    readings: pd.DataFrame, sections: pd.DataFrame, thresholds: Thresholds, nominal_temperature_c: float
) -> pd.DataFrame:
    """Apply every rule to every section and return a table of events."""
    events: list[dict] = []
    last_ts = readings["timestamp"].max()
    expected = sections.set_index("section_id")["expected_pressure_kpa"]
    expected_t = _expected_temperature(sections, nominal_temperature_c)

    for sid, df in readings.groupby("section_id", sort=True):
        df = df.sort_values("timestamp").reset_index(drop=True)
        ts = df["timestamp"]
        p = df["pressure_kpa"].to_numpy()
        exp_p = float(expected[sid])
        p_dev_pct = 100 * (p - exp_p) / exp_p
        p_rate_pct = 100 * np.abs(np.diff(p, prepend=p[0])) / exp_p
        exp_t = float(expected_t[sid])
        t_dev = df["temperature_c"].to_numpy() - exp_t
        imbalance = df["flow_imbalance_pct"].to_numpy()
        health = df["sensor_health"].to_numpy()

        # Stuck sensor: the last N pressure readings are all identical.
        w = thresholds.stuck_window_readings
        # (NaN-tolerant, so occasional missing values don't hide a frozen sensor.)
        roll = pd.Series(p).rolling(w, min_periods=max(2, w // 2))
        rolling_range = (roll.max() - roll.min()).to_numpy()
        stuck = np.nan_to_num(rolling_range, nan=1.0) == 0.0

        with np.errstate(invalid="ignore"):
            rules = [
                # (rule name, category, parameter, mask, values, threshold, base severity, min run, explain fn)
                ("High pressure", "Process", "Pressure", p_dev_pct > thresholds.pressure_high_pct, p_dev_pct,
                 thresholds.pressure_high_pct, "High", 1,
                 lambda peak, thr: f"Pressure reached {peak:+.1f}% versus the section's expected "
                                   f"{exp_p:,.0f} kPa (limit +{thr:.1f}%)."),
                ("Low pressure", "Process", "Pressure", p_dev_pct < -thresholds.pressure_low_pct, -p_dev_pct,
                 thresholds.pressure_low_pct, "Medium", 1,
                 lambda peak, thr: f"Pressure fell {peak:.1f}% below the section's expected "
                                   f"{exp_p:,.0f} kPa (limit -{thr:.1f}%)."),
                ("Rapid pressure change", "Process", "Pressure", p_rate_pct > thresholds.pressure_rate_pct, p_rate_pct,
                 thresholds.pressure_rate_pct, "Medium", 1,
                 lambda peak, thr: f"Pressure changed by up to {peak:.1f}% between consecutive readings "
                                   f"(limit {thr:.1f}%)."),
                ("Temperature deviation", "Process", "Temperature",
                 np.abs(t_dev) > thresholds.temperature_deviation_c, np.abs(t_dev),
                 thresholds.temperature_deviation_c, "Medium", 1,
                 lambda peak, thr: f"Temperature was up to {peak:.1f} °C away from the section's expected "
                                   f"{exp_t:.0f} °C (limit {thr:.0f} °C)."),
                ("Possible leak indicator", "Process", "Flow balance",
                 imbalance > thresholds.flow_imbalance_pct, imbalance,
                 thresholds.flow_imbalance_pct, "High", thresholds.flow_imbalance_min_readings,
                 lambda peak, thr: f"Outlet flow was up to {peak:.1f}% lower than inlet flow for a sustained "
                                   f"period (limit {thr:.1f}%). Simulated indicator only; requires field verification."),
                ("Sensor health low", "Sensor", "Sensor health", health < thresholds.sensor_health_min,
                 thresholds.sensor_health_min - health, thresholds.sensor_health_min, "Low", 1,
                 lambda peak, thr: f"Sensor self-reported health dropped to {thr - peak:.0f}/100 "
                                   f"(minimum {thr:.0f})."),
                ("Sensor possibly stuck", "Sensor", "Pressure", stuck, rolling_range,
                 0.0, "Medium", 1,
                 lambda peak, thr: f"Pressure reading did not change for at least {w} consecutive readings; "
                                   f"the transmitter may be frozen, so this section's pressure is unreliable."),
                ("Missing readings", "Sensor", "Pressure / temperature",
                 df[["pressure_kpa", "temperature_c"]].isna().any(axis=1).to_numpy(), np.ones(len(df)),
                 0.0, "Low", 1,
                 lambda peak, thr: "One or more readings were missing (no value received)."),
            ]

        for name, category, parameter, mask, values, thr, base_sev, min_run, explain in rules:
            mask = np.asarray(mask, dtype=bool)
            for start, end in _runs(mask, thresholds.merge_gap_readings):
                if mask[start : end + 1].sum() < min_run:
                    continue
                window = values[start : end + 1]
                peak = float(np.nanmax(window)) if np.isfinite(window).any() else float("nan")
                ratio = peak / thr if thr > 0 and np.isfinite(peak) else 1.0
                severity = _escalate(base_sev, ratio) if category == "Process" else base_sev
                events.append(
                    {
                        "section_id": sid,
                        "rule": name,
                        "category": category,
                        "parameter": parameter,
                        "severity": severity,
                        "priority": SEVERITY_PRIORITY[severity],
                        "start": ts.iloc[start],
                        "end": ts.iloc[end],
                        "duration_min": (ts.iloc[end] - ts.iloc[start]).total_seconds() / 60,
                        "peak_value": peak,
                        "threshold": thr,
                        "active": ts.iloc[end] == last_ts,
                        "explanation": explain(peak, thr),
                        "detector": RULES_DETECTOR,
                    }
                )

    if not events:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    out = pd.DataFrame(events).sort_values(["priority", "start"], ascending=[False, True]).reset_index(drop=True)
    out.insert(0, "event_id", [f"EV-{i + 1:03d}" for i in range(len(out))])
    return add_leak_evidence(out[EVENT_COLUMNS], readings, sections)


def add_leak_evidence(events: pd.DataFrame, readings: pd.DataFrame, sections: pd.DataFrame,
                      drop_pct: float = 1.0) -> pd.DataFrame:
    """Cross-check each flow-imbalance alert against downstream pressure.

    A real loss of fluid usually lowers pressure downstream as well. A flow imbalance with no
    matching pressure change points more towards a meter problem. This is supporting evidence
    for the operator, not a diagnosis.
    """
    out = events.copy()
    expected = sections.set_index("section_id")["expected_pressure_kpa"]
    order = sections["section_id"].tolist()
    for i, ev in out[out["rule"] == "Possible leak indicator"].iterrows():
        downstream = order[order.index(ev["section_id"]) + 1:]
        if not downstream:
            note = " Cross-check: no downstream pressure sensor available."
        else:
            d = readings[readings["section_id"].isin(downstream)].copy()
            d["dev"] = 100 * (d["pressure_kpa"] - d["section_id"].map(expected)) / d["section_id"].map(expected)
            before = d[(d["timestamp"] < ev["start"]) & (d["timestamp"] >= ev["start"] - pd.Timedelta(hours=6))]
            during = d[(d["timestamp"] >= ev["start"]) & (d["timestamp"] <= ev["end"])]
            change = during["dev"].mean() - before["dev"].mean() if len(before) and len(during) else float("nan")
            if pd.isna(change):
                note = " Cross-check: not enough pressure history to compare."
            elif change <= -drop_pct:
                note = (f" Cross-check: downstream pressure fell by {abs(change):.1f}% at the same time, which is "
                        "consistent with a loss of fluid.")
            else:
                note = (f" Cross-check: no matching downstream pressure drop ({change:+.1f}%), so a flow-meter "
                        "fault is also possible.")
        out.at[i, "explanation"] = ev["explanation"] + note
    return out


def events_as_of(readings: pd.DataFrame, sections: pd.DataFrame, thresholds: Thresholds,
                 nominal_temperature_c: float, timestamp: pd.Timestamp) -> pd.DataFrame:
    """Events exactly as they would have been known at `timestamp` (no look-ahead).

    Used for replay: only readings up to that moment are used, so durations, peaks
    and explanations never include future data.
    """
    return detect_events(readings[readings["timestamp"] <= timestamp], sections, thresholds, nominal_temperature_c)


def events_at(events: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
    """Events in progress at the given time, from a table of already-detected events."""
    if events.empty:
        return events
    return events[(events["start"] <= timestamp) & (events["end"] >= timestamp)]


def section_status(sections: pd.DataFrame, current_events: pd.DataFrame) -> pd.DataFrame:
    """Summarise each section into one status label based on its current events.

    Alert = Critical/High process event; Watch = Medium process event;
    Sensor issue = only instrument problems; Advisory = only Low process events or
    machine-learning flags (ML never raises a section above Advisory on its own).
    """
    rows = []
    for sid in sections["section_id"]:
        ev = current_events[current_events["section_id"] == sid] if not current_events.empty else current_events
        process = ev[ev["category"] == "Process"] if not ev.empty else ev
        if not process.empty and process["priority"].max() >= SEVERITY_PRIORITY["High"]:
            status = "Alert"
        elif not process.empty and process["priority"].max() == SEVERITY_PRIORITY["Medium"]:
            status = "Watch"
        elif not ev.empty and (ev["category"] == "Sensor").any():
            status = "Sensor issue"
        elif not process.empty or (not ev.empty and (ev["category"] == "ML").any()):
            status = "Advisory"
        else:
            status = "Normal"
        rows.append(
            {
                "section_id": sid,
                "status": status,
                "active_events": 0 if ev.empty else len(ev),
                "reasons": "; ".join(ev["rule"].unique()) if not ev.empty else "",
            }
        )
    return pd.DataFrame(rows)
