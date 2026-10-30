"""Simulated pipeline sensor readings.

This module generates SYNTHETIC time series for pressure, temperature, flow and
sensor health along a pipeline divided into sections. Scenarios inject events
(possible leak, pressure surge, sensor fault, temperature excursion) so the
alerting logic and dashboards can be demonstrated.

The physics is deliberately simple (linear pressure loss, daily cycles, random
noise). It is a teaching and demonstration model, not a hydraulic simulator.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from modules.config import PIPELINE_PROFILES, PipelineProfile

SCENARIOS: dict[str, str] = {
    "Normal operations": "No injected events. Readings vary with daily demand cycles and sensor noise.",
    "Possible leak (flow imbalance)": (
        "From the onset time, part of the flow is lost inside the affected section, so the "
        "section's outlet flow is lower than its inlet flow and downstream pressure falls."
    ),
    "Pressure surge": (
        "A short pressure transient (for example, from a fast valve closure) starts in the "
        "affected section and spreads, weakened, to neighbouring sections."
    ),
    "Sensor fault": (
        "The affected section's pressure transmitter freezes at its last value, some readings "
        "go missing, and its self-reported health declines."
    ),
    "Temperature excursion": (
        "Fluid temperature in the affected section rises steadily over several hours."
    ),
    "Flow meter fault (bias)": (
        "The affected section's outlet flow meter starts reading low. Flow balance looks like a leak, but "
        "pressures do not change: a classic false-alarm pattern. The meter still reports itself healthy."
    ),
    "Pressure sensor drift": (
        "The affected section's pressure transmitter slowly drifts upward over a day while still reporting "
        "itself healthy. Simple limits catch it only once the drift is large."
    ),
}

# Which sections an injected event really affects, and for how long (hours after onset; None = until the end).
# Used only to evaluate detectors against the simulation's own "ground truth".
_TRUTH_WINDOWS = {
    "Possible leak (flow imbalance)": (0, None, "Process"),
    "Pressure surge": (0, 1.5, "Process"),
    "Sensor fault": (0, None, "Instrument"),
    "Temperature excursion": (0, None, "Process"),
    "Flow meter fault (bias)": (0, None, "Instrument"),
    "Pressure sensor drift": (0, None, "Instrument"),
}


@dataclass(frozen=True)
class SimulationConfig:
    """All settings that control one simulation run."""

    pipeline_type: str = "Oil"
    pipeline_length_km: float = 120.0
    n_sections: int = 8
    duration_hours: int = 72
    interval_minutes: int = 15
    scenario: str = "Normal operations"
    affected_section: int = 4  # 1-based section number
    onset_hour: float = 48.0  # hours after the start of the simulated window
    severity: float = 0.5  # 0.1 (mild) to 1.0 (severe)
    noise_level: float = 1.0  # multiplies all random noise
    seed: int = 42
    end_time: str = "2026-01-15 00:00"  # fixed so runs are reproducible

    @property
    def profile(self) -> PipelineProfile:
        return PIPELINE_PROFILES[self.pipeline_type]

    @property
    def n_steps(self) -> int:
        return int(self.duration_hours * 60 / self.interval_minutes) + 1


def section_id(number: int) -> str:
    """Section numbers are 1-based; IDs look like 'S01'."""
    return f"S{number:02d}"


def expected_pressure_kpa(profile, km):
    """Linear friction loss, restored to the inlet pressure at each pump/compressor station."""
    return profile.nominal_pressure_kpa - profile.pressure_drop_per_km * (np.asarray(km) % profile.station_spacing_km)


def expected_temperature_c(profile, km):
    """Cooling towards ground temperature (about 0.03 °C/km at first, levelling off on long lines)."""
    gap = profile.nominal_temperature_c - profile.ground_temperature_c
    return profile.ground_temperature_c + gap * np.exp(-0.03 * np.asarray(km) / gap)


def build_sections(config: SimulationConfig) -> pd.DataFrame:
    """Split the pipeline into equal-length sections, each with one sensor station."""
    edges = np.linspace(0.0, config.pipeline_length_km, config.n_sections + 1)
    mid = (edges[:-1] + edges[1:]) / 2
    profile = config.profile
    return pd.DataFrame(
        {
            "section_id": [section_id(i + 1) for i in range(config.n_sections)],
            "section_number": np.arange(1, config.n_sections + 1),
            "start_km": edges[:-1],
            "end_km": edges[1:],
            "sensor_km": mid,
            # The pressure and temperature we expect at this sensor under normal conditions.
            "expected_pressure_kpa": expected_pressure_kpa(profile, mid),
            "expected_temperature_c": expected_temperature_c(profile, mid),
        }
    )


def _ramp(hours: np.ndarray, onset: float, ramp_hours: float) -> np.ndarray:
    """0 before onset, rising linearly to 1 over ramp_hours, then staying at 1."""
    return np.clip((hours - onset) / ramp_hours, 0.0, 1.0)


def simulate_readings(config: SimulationConfig) -> pd.DataFrame:
    """Generate synthetic readings for every section at every time step.

    Returns a "long" table with one row per (timestamp, section).
    """
    if config.scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {config.scenario}")
    if not 1 <= config.affected_section <= config.n_sections:
        raise ValueError("affected_section must be between 1 and n_sections")

    rng = np.random.default_rng(config.seed)
    profile = config.profile
    sections = build_sections(config)
    n_t, n_s = config.n_steps, config.n_sections
    noise = config.noise_level
    k = config.affected_section - 1  # 0-based index of the affected section

    hours = np.arange(n_t) * config.interval_minutes / 60.0
    h = hours[:, None]  # shape (time, 1) so it broadcasts across sections
    daily = np.sin(2 * np.pi * (h - 6) / 24.0)  # demand peaks mid-afternoon
    expected_p = sections["expected_pressure_kpa"].to_numpy()[None, :]
    expected_t = sections["expected_temperature_c"].to_numpy()[None, :]

    # --- Flow: the same physical flow passes through every section; each section
    # has an inlet and an outlet meter with independent measurement noise.
    true_flow = profile.nominal_flow * (1 + 0.04 * daily) * np.ones((n_t, n_s))
    leak_loss = np.zeros((n_t, n_s))

    # --- Pressure: linear loss along the line plus a small daily swing.
    pressure = expected_p * (1 + 0.01 * daily)

    # --- Temperature: daily swing and slow cooling along the line.
    temperature = expected_t + 2.0 * daily
    temperature = temperature * np.ones((n_t, n_s))

    # --- Sensor health (0-100): a slow random walk in the 93-100 range.
    health_walk = np.cumsum(rng.normal(0, 0.15 * noise, (n_t, n_s)), axis=0)
    health = np.clip(97.5 + health_walk, 93.0, 100.0)

    sev = config.severity
    onset = config.onset_hour

    if config.scenario == "Possible leak (flow imbalance)":
        leak = sev * 0.15 * profile.nominal_flow * _ramp(hours, onset, 1.0)
        leak_loss[:, k] = leak
        # Everything downstream of the leak carries less flow.
        true_flow[:, k + 1 :] -= leak[:, None]
        pressure_drop = sev * 0.08 * expected_p[0, :] * _ramp(hours, onset, 1.0)[:, None]
        pressure[:, k + 1 :] -= pressure_drop[:, k + 1 :]
        pressure[:, k] -= 0.5 * pressure_drop[:, k]

    elif config.scenario == "Pressure surge":
        dt = h - onset
        pulse = np.where(dt >= 0, np.exp(-dt / 0.25), 0.0)  # decays over ~45 min
        distance = np.abs(np.arange(n_s) - k)[None, :]
        pressure += sev * 0.30 * expected_p * pulse * 0.5**distance

    elif config.scenario == "Temperature excursion":
        temperature[:, k] += sev * 28.0 * _ramp(hours, onset, 6.0)

    elif config.scenario == "Pressure sensor drift":
        # Measurement error only: the real pressure is unchanged.
        pressure[:, k] += sev * 0.15 * expected_p[0, k] * _ramp(hours, onset, 24.0)

    # Each section's inlet meter sees the flow entering it; the outlet meter sees
    # that flow minus anything lost inside the section (non-zero only for a leak).
    flow_in = true_flow
    flow_out = true_flow - leak_loss
    if config.scenario == "Flow meter fault (bias)":
        flow_out = flow_out.copy()
        flow_out[:, k] *= 1 - sev * 0.12 * _ramp(hours, onset, 0.5)

    # Measurement noise is added last, on top of the "true" process values.
    flow_in = flow_in * (1 + rng.normal(0, 0.003 * noise, (n_t, n_s)))
    flow_out = flow_out * (1 + rng.normal(0, 0.003 * noise, (n_t, n_s)))
    pressure = pressure * (1 + rng.normal(0, 0.003 * noise, (n_t, n_s)))
    temperature = temperature + rng.normal(0, 0.2 * noise, (n_t, n_s))

    if config.scenario == "Sensor fault":
        fault = hours >= onset
        first = int(np.argmax(fault))
        pressure[fault, k] = pressure[first, k]  # transmitter frozen
        health[:, k] -= sev * 70.0 * _ramp(hours, onset, 4.0)
        dropouts = fault & (rng.random(n_t) < 0.08 * sev)
        pressure[dropouts, k] = np.nan
        temperature[dropouts, k] = np.nan

    timestamps = pd.date_range(end=pd.Timestamp(config.end_time), periods=n_t, freq=f"{config.interval_minutes}min")
    frame = pd.DataFrame(
        {
            "timestamp": np.repeat(timestamps, n_s),
            "hour": np.repeat(hours, n_s),
            "section_id": np.tile(sections["section_id"].to_numpy(), n_t),
            "pressure_kpa": pressure.ravel(),
            "temperature_c": temperature.ravel(),
            "flow_in": flow_in.ravel(),
            "flow_out": flow_out.ravel(),
            "sensor_health": np.clip(health, 0, 100).ravel(),
        }
    )
    frame["flow_imbalance_pct"] = 100 * (frame["flow_in"] - frame["flow_out"]) / frame["flow_in"]
    return frame


def injected_truth(config: SimulationConfig) -> pd.DataFrame:
    """Ground truth of the simulation: where and when the injected event really is.

    Returns one row per affected section with start/end timestamps and the kind of
    event ("Process" or "Instrument"). Empty for normal operations.
    """
    columns = ["section_id", "start", "end", "kind"]
    if config.scenario not in _TRUTH_WINDOWS:
        return pd.DataFrame(columns=columns)
    begin, length, kind = _TRUTH_WINDOWS[config.scenario]
    end_time = pd.Timestamp(config.end_time)
    window_start = end_time - pd.Timedelta(hours=config.duration_hours)
    start = window_start + pd.Timedelta(hours=config.onset_hour + begin)
    end = end_time if length is None else min(end_time, start + pd.Timedelta(hours=length))
    affected = [config.affected_section]
    if config.scenario == "Possible leak (flow imbalance)":  # pressure also falls downstream
        affected += list(range(config.affected_section + 1, config.n_sections + 1))
    if config.scenario == "Pressure surge":  # the transient also reaches the neighbours
        affected += [s for s in (config.affected_section - 1, config.affected_section + 1) if 1 <= s <= config.n_sections]
    return pd.DataFrame([{"section_id": section_id(s), "start": start, "end": end, "kind": kind} for s in affected],
                        columns=columns)


def readings_at(readings: pd.DataFrame, timestamp: pd.Timestamp) -> pd.DataFrame:
    """Return one row per section for the given timestamp."""
    return readings[readings["timestamp"] == timestamp].reset_index(drop=True)
