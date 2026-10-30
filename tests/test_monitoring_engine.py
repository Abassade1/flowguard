"""Tests for the synthetic sensor simulation."""

import numpy as np
import pytest

from modules.monitoring_engine import SCENARIOS, SimulationConfig, build_sections, simulate_readings


def test_sections_cover_whole_pipeline():
    config = SimulationConfig(pipeline_length_km=100, n_sections=5)
    sections = build_sections(config)
    assert len(sections) == 5
    assert sections["start_km"].iloc[0] == 0
    assert sections["end_km"].iloc[-1] == pytest.approx(100)
    # Pressure falls with distance from the inlet.
    assert sections["expected_pressure_kpa"].is_monotonic_decreasing


def test_shape_and_reproducibility():
    config = SimulationConfig(duration_hours=24, n_sections=6)
    a, b = simulate_readings(config), simulate_readings(config)
    assert len(a) == config.n_steps * 6
    assert a.equals(b)  # same seed -> identical data


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_every_scenario_runs(scenario):
    readings = simulate_readings(SimulationConfig(scenario=scenario))
    assert {"pressure_kpa", "temperature_c", "flow_in", "flow_out", "sensor_health"} <= set(readings.columns)
    assert readings["sensor_health"].between(0, 100).all()


def test_leak_creates_imbalance_only_in_affected_section_after_onset():
    config = SimulationConfig(scenario="Possible leak (flow imbalance)", affected_section=3, onset_hour=24,
                              severity=1.0, duration_hours=48)
    r = simulate_readings(config)
    after = r[r["hour"] > 26]
    mean_imb = after.groupby("section_id")["flow_imbalance_pct"].mean()
    assert mean_imb["S03"] > 10
    assert mean_imb.drop("S03").abs().max() < 1
    before = r[(r["hour"] < 23) & (r["section_id"] == "S03")]
    assert before["flow_imbalance_pct"].abs().mean() < 1


def test_sensor_fault_freezes_pressure():
    config = SimulationConfig(scenario="Sensor fault", affected_section=2, onset_hour=10, duration_hours=24)
    r = simulate_readings(config)
    frozen = r[(r["section_id"] == "S02") & (r["hour"] >= 10)]["pressure_kpa"].dropna()
    assert np.ptp(frozen) == 0


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        simulate_readings(SimulationConfig(scenario="Nope"))
    with pytest.raises(ValueError):
        simulate_readings(SimulationConfig(n_sections=4, affected_section=9))
