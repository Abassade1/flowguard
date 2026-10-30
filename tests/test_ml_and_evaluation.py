"""Tests for Phase 6: causal replay, leak cross-check, ML detection, evaluation and acknowledgements."""

import pandas as pd
import pytest

from modules import alert_acks as acks
from modules.anomaly_detection import Thresholds, detect_events, events_as_of, section_status
from modules.detector_evaluation import compare_detectors, evaluate
from modules.ml_detection import ML_DETECTOR, MLSettings, build_features, ml_events, run_ml
from modules.monitoring_engine import SimulationConfig, build_sections, injected_truth, simulate_readings


def setup(**kwargs):
    c = SimulationConfig(**kwargs)
    return c, build_sections(c), simulate_readings(c)


def test_events_as_of_never_uses_future_data():
    c, s, r = setup(scenario="Possible leak (flow imbalance)", onset_hour=48)
    early = pd.Timestamp(r["timestamp"].min()) + pd.Timedelta(hours=50)
    known = events_as_of(r, s, Thresholds(), 35, early)
    assert not known.empty
    assert (known["end"] <= early).all()
    full = detect_events(r, s, Thresholds(), 35)
    assert known["duration_min"].max() < full["duration_min"].max()


def test_leak_cross_check_separates_leak_from_meter_fault():
    for scenario, phrase in [("Possible leak (flow imbalance)", "consistent with a loss of fluid"),
                             ("Flow meter fault (bias)", "flow-meter fault is also possible")]:
        c, s, r = setup(scenario=scenario, severity=0.6)
        ev = detect_events(r, s, Thresholds(), 35)
        leak = ev[ev["rule"] == "Possible leak indicator"]
        assert len(leak) == 1 and phrase in leak["explanation"].iloc[0], scenario


def test_truth_windows():
    assert injected_truth(SimulationConfig()).empty
    leak = injected_truth(SimulationConfig(scenario="Possible leak (flow imbalance)", affected_section=6))
    assert list(leak["section_id"]) == ["S06", "S07", "S08"]  # pressure also falls downstream
    surge = injected_truth(SimulationConfig(scenario="Pressure surge", affected_section=1))
    assert list(surge["section_id"]) == ["S01", "S02"]
    assert (surge["end"] - surge["start"]).iloc[0] == pd.Timedelta(hours=1.5)


def test_features_are_backward_looking():
    c, s, r = setup()
    f_full = build_features(r, s, 35)
    cut = r["timestamp"].min() + pd.Timedelta(hours=10)
    part = r[r["timestamp"] <= cut]
    f_part = build_features(part, s, 35)
    pd.testing.assert_frame_equal(f_full.loc[part.index], f_part, check_exact=False)


@pytest.mark.parametrize("seed", [1, 2])
def test_ml_quiet_in_normal_operations(seed):
    c, s, r = setup(seed=seed)
    assert ml_events(run_ml(r, s, 35)).empty


def test_ml_catches_drift_that_rules_miss():
    c, s, r = setup(scenario="Pressure sensor drift", severity=0.6)
    assert detect_events(r, s, Thresholds(), 35).empty
    ml = ml_events(run_ml(r, s, 35))
    assert (ml["section_id"] == "S04").any()
    assert (ml["detector"] == ML_DETECTOR).all() and (ml["category"] == "ML").all()
    assert ml["explanation"].str.contains("pressure vs expected").any()


def test_ml_events_respect_as_of_and_baseline():
    c, s, r = setup(scenario="Possible leak (flow imbalance)")
    res = run_ml(r, s, 35, MLSettings(baseline_hours=24))
    assert (ml_events(res)["start"] >= res.baseline_end).all()
    cut = res.baseline_end + pd.Timedelta(hours=12)  # before the leak at hour 48
    assert ml_events(res, as_of=cut).empty


def test_isolation_forest_option_runs():
    c, s, r = setup(scenario="Possible leak (flow imbalance)")
    res = run_ml(r, s, 35, MLSettings(model="isolation_forest"))
    assert res.scores["score"].notna().all()


def test_ml_alone_never_raises_above_advisory():
    c, s, r = setup(scenario="Pressure sensor drift", severity=0.6)
    ml = ml_events(run_ml(r, s, 35))
    status = section_status(s, ml[ml["active"]])
    assert set(status["status"]) <= {"Normal", "Advisory"}
    assert status.set_index("section_id").loc["S04", "status"] == "Advisory"


def test_baseline_too_short_is_rejected():
    c, s, r = setup()
    with pytest.raises(ValueError):
        run_ml(r, s, 35, MLSettings(baseline_hours=0.5))


def test_evaluation_metrics():
    c, s, r = setup(scenario="Possible leak (flow imbalance)")
    truth = injected_truth(c)
    rules = detect_events(r, s, Thresholds(), 35)
    result = evaluate(rules, truth)
    assert result["detected"] is True and 0 <= result["delay_hours"] < 2 and result["false_alarms"] == 0
    normal = evaluate(rules.iloc[0:0], injected_truth(SimulationConfig()))
    assert normal["detected"] is None and normal["false_alarms"] == 0
    table = compare_detectors(rules, ml_events(run_ml(r, s, 35)), truth)
    assert list(table["detector"]) == ["Rules", "ML (experimental)", "Either (rules or ML)"]


def test_acknowledgements_round_trip(tmp_path):
    db = tmp_path / "t.db"
    c, s, r = setup(scenario="Possible leak (flow imbalance)")
    ev = detect_events(r, s, Thresholds(), 35)
    key = acks.config_key(c)
    assert key == acks.config_key(SimulationConfig(scenario="Possible leak (flow imbalance)"))
    shown = acks.with_acks(ev, key, acks.acks_for(key, db))
    assert (shown["ack_status"] == "New").all()
    acks.set_ack(shown["ack_key"].iloc[0], "Under investigation", "Crew dispatched (simulated)", "Ops", db)
    shown = acks.with_acks(ev, key, acks.acks_for(key, db))
    assert shown["ack_status"].iloc[0] == "Under investigation"
    # The key is stable when the same event is re-detected later in the replay.
    later = events_as_of(r, s, Thresholds(), 35, r["timestamp"].max() - pd.Timedelta(hours=1))
    assert acks.with_acks(later, key, acks.acks_for(key, db))["ack_status"].iloc[0] == "Under investigation"
    acks.set_ack(shown["ack_key"].iloc[0], "New", db_path=db)
    assert acks.acks_for(key, db) == {}
    with pytest.raises(ValueError):
        acks.set_ack("x", "Ignored", db_path=db)
