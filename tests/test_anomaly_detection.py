"""Tests for the rule-based alerting."""

import numpy as np

from modules.anomaly_detection import Thresholds, _runs, detect_events, events_at, section_status
from modules.monitoring_engine import SimulationConfig, build_sections, simulate_readings


def run(**kwargs):
    config = SimulationConfig(**kwargs)
    sections = build_sections(config)
    readings = simulate_readings(config)
    events = detect_events(readings, sections, Thresholds(), config.profile.nominal_temperature_c)
    return sections, readings, events


def test_runs_merges_small_gaps():
    mask = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1], dtype=bool)
    assert _runs(mask, merge_gap=1) == [(1, 4), (8, 8)]
    assert _runs(mask, merge_gap=0) == [(1, 2), (4, 4), (8, 8)]
    assert _runs(np.zeros(5, dtype=bool), 2) == []


def test_normal_operations_raise_no_alerts():
    for pipeline_type in ["Oil", "Natural gas", "Water"]:
        _, _, events = run(pipeline_type=pipeline_type)
        assert events.empty, events


def test_leak_is_flagged_in_the_right_section():
    _, _, events = run(scenario="Possible leak (flow imbalance)", affected_section=5)
    leak = events[events["rule"] == "Possible leak indicator"]
    assert set(leak["section_id"]) == {"S05"}
    assert leak["active"].all()
    assert "field verification" in leak["explanation"].iloc[0]


def test_surge_flags_high_pressure():
    _, _, events = run(scenario="Pressure surge", affected_section=2, severity=1.0)
    assert ((events["rule"] == "High pressure") & (events["section_id"] == "S02")).any()


def test_sensor_fault_is_classified_as_sensor_issue():
    sections, readings, events = run(scenario="Sensor fault", affected_section=3, severity=1.0)
    assert {"Sensor possibly stuck", "Sensor health low"} <= set(events["rule"])
    status = section_status(sections, events_at(events, readings["timestamp"].max()))
    assert status.set_index("section_id").loc["S03", "status"] == "Sensor issue"


def test_mild_leak_can_go_undetected():
    # Honest behaviour: a very small imbalance stays below the demonstration threshold.
    _, _, events = run(scenario="Possible leak (flow imbalance)", severity=0.1)
    assert "Possible leak indicator" not in set(events["rule"])


def test_status_is_alert_for_high_priority_event():
    sections, readings, events = run(scenario="Possible leak (flow imbalance)", affected_section=4)
    status = section_status(sections, events_at(events, readings["timestamp"].max()))
    s = status.set_index("section_id")["status"]
    assert s["S04"] == "Alert"
    assert (s.drop("S04") == "Normal").all()
