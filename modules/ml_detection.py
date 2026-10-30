"""Optional machine-learning anomaly detection (scikit-learn).

What it does
------------
1. Turns each reading into a few interpretable features (pressure deviation, rate of
   change, temperature deviation, flow imbalance, sensor health, short-term pressure
   variability, missing data).
2. Learns what "normal" looks like from a BASELINE period at the start of the window
   (assumed normal; if an event starts inside the baseline, the model learns it as normal).
   Two unsupervised models are available:
   * Robust covariance (EllipticEnvelope): distance from the normal pattern, which keeps
     growing as a reading moves further away, so slow drifts are visible (default).
   * Isolation Forest: good at odd combinations, but its score stops growing once a value is
     outside the range seen in the baseline, so it can miss how far a slow drift has gone.
3. Scores every reading. Readings more unusual than almost everything in the baseline
   (above a chosen quantile of baseline scores) are flagged; consecutive flags become events.
4. Explains each event by the features that departed most from the baseline (in standard
   deviations). This is an approximation: neither model gives exact reasons.

What it is NOT
--------------
* Not a leak detector, not validated, not for real operations. It runs on simulated data.
* It never raises a section above "Advisory" by itself and never triggers any action.
  The transparent rules remain the primary alerting method.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest

from modules.anomaly_detection import EVENT_COLUMNS, _runs

ML_DETECTOR = "ML (experimental)"
MODELS = {"robust_covariance": "Robust covariance (EllipticEnvelope)", "isolation_forest": "Isolation Forest"}

FEATURE_LABELS = {
    "pressure_dev_pct": "pressure vs expected",
    "pressure_rate_pct": "pressure change per reading",
    "temperature_dev_c": "temperature vs nominal",
    "flow_imbalance_pct": "flow imbalance",
    "sensor_health": "sensor health",
    "pressure_variability": "short-term pressure variability",
    "missing": "missing readings",
}
FEATURES = list(FEATURE_LABELS)


@dataclass(frozen=True)
class MLSettings:
    model: str = "robust_covariance"  # key of MODELS
    baseline_hours: float = 24.0  # learn "normal" from this many hours at the start of the window
    sensitivity_quantile: float = 0.999  # reference point: this quantile of baseline scores...
    threshold_margin: float = 1.1  # ...times this margin is the alert threshold (higher = fewer alerts)
    min_consecutive: int = 3  # a flag must persist this many readings to become an event
    n_estimators: int = 200
    window_readings: int = 8  # for short-term variability
    seed: int = 7


@dataclass
class MLResult:
    scores: pd.DataFrame  # timestamp, section_id, score, flagged, z_<feature> ...
    threshold: float
    baseline_end: pd.Timestamp
    baseline_rows: int
    notes: list[str]


def build_features(readings: pd.DataFrame, sections: pd.DataFrame, nominal_temperature_c: float,
                   window: int = 8) -> pd.DataFrame:
    """One row of features per reading (same order as `readings`)."""
    expected = readings["section_id"].map(sections.set_index("section_id")["expected_pressure_kpa"])
    out = pd.DataFrame(index=readings.index)
    out["pressure_dev_pct"] = 100 * (readings["pressure_kpa"] - expected) / expected
    grouped = readings.groupby("section_id")["pressure_kpa"]
    out["pressure_rate_pct"] = 100 * grouped.diff().abs() / expected
    out["temperature_dev_c"] = readings["temperature_c"] - nominal_temperature_c
    out["flow_imbalance_pct"] = readings["flow_imbalance_pct"]
    out["sensor_health"] = readings["sensor_health"]
    # Backward-looking rolling standard deviation (uses only past readings).
    # (Scaled by the section's expected pressure, which is known in advance: no future data is used.)
    rolling_std = grouped.transform(lambda s: s.rolling(window, min_periods=max(2, window // 2)).std())
    out["pressure_variability"] = 100 * rolling_std / expected
    out["missing"] = readings[["pressure_kpa", "temperature_c"]].isna().any(axis=1).astype(float)
    return out[FEATURES]


def run_ml(readings: pd.DataFrame, sections: pd.DataFrame, nominal_temperature_c: float,
           settings: MLSettings = MLSettings()) -> MLResult:
    """Train on the baseline period and score every reading."""
    feats = build_features(readings, sections, nominal_temperature_c, settings.window_readings)
    start = readings["timestamp"].min()
    baseline_end = start + pd.Timedelta(hours=settings.baseline_hours)
    is_base = (readings["timestamp"] < baseline_end).to_numpy()
    notes = []
    if is_base.sum() < 50:
        raise ValueError("The baseline period is too short to learn normal behaviour.")

    base = feats[is_base]
    fill = base.median()
    # Sensor health wanders slowly in normal operation, so it gets a wide floor.
    floors = pd.Series({"pressure_dev_pct": 0.3, "pressure_rate_pct": 0.2, "temperature_dev_c": 0.3,
                        "flow_imbalance_pct": 0.15, "sensor_health": 3.0, "pressure_variability": 0.15,
                        "missing": 0.1})
    filled = feats.fillna(fill)
    mean, std = filled[is_base].mean(), filled[is_base].std().clip(lower=floors)
    z = (filled - mean) / std

    # Features that never vary in the baseline (e.g. no missing readings) cannot define "normal" for a
    # covariance model; they are left to the rules and still shown in explanations.
    used = [f for f in FEATURES if base[f].std(skipna=True) > 0] if settings.model == "robust_covariance" else FEATURES
    if settings.model == "robust_covariance":
        model = EllipticEnvelope(support_fraction=0.9, random_state=settings.seed)
    else:
        model = IsolationForest(n_estimators=settings.n_estimators, contamination="auto", random_state=settings.seed)
    model.fit(z.loc[is_base, used].to_numpy())
    score = -model.score_samples(z[used].to_numpy())  # higher = more unusual
    if settings.model == "robust_covariance":
        score = np.sqrt(np.maximum(score, 0))  # squared Mahalanobis distance -> distance in SD units
    threshold = float(np.quantile(score[is_base], settings.sensitivity_quantile)) * settings.threshold_margin

    scores = pd.DataFrame({"timestamp": readings["timestamp"].to_numpy(), "section_id": readings["section_id"].to_numpy(),
                           "score": score, "flagged": score > threshold})
    for f in FEATURES:
        scores[f"z_{f}"] = z[f].to_numpy()
    notes.append(f"{MODELS[settings.model]} trained on {int(is_base.sum())} readings from the first "
                 f"{settings.baseline_hours:g} hours, assumed to be normal operation.")
    if len(used) < len(FEATURES):
        notes.append("Not used by the model (no variation in the baseline): "
                     + ", ".join(FEATURE_LABELS[f] for f in FEATURES if f not in used) + ".")
    return MLResult(scores, threshold, baseline_end, int(is_base.sum()), notes)


def ml_events(result: MLResult, settings: MLSettings = MLSettings(),
              as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Group flagged readings into events, using only scores up to `as_of` (no look-ahead)."""
    s = result.scores if as_of is None else result.scores[result.scores["timestamp"] <= as_of]
    last_ts = s["timestamp"].max()
    rows = []
    for sid, df in s.groupby("section_id", sort=True):
        df = df.sort_values("timestamp").reset_index(drop=True)
        df = df[df["timestamp"] >= result.baseline_end].reset_index(drop=True)  # judge only after the baseline
        if df.empty:
            continue
        mask = df["flagged"].to_numpy()
        for a, b in _runs(mask, merge_gap=1):
            if mask[a:b + 1].sum() < settings.min_consecutive:
                continue
            window = df.iloc[a:b + 1]
            peak = window.loc[window["score"].idxmax()]
            zs = {f: peak[f"z_{f}"] for f in FEATURES}
            top = sorted(zs, key=lambda f: abs(zs[f]), reverse=True)[:2]
            signals = "; ".join(f"{FEATURE_LABELS[f]} {zs[f]:+.1f} SD from normal" for f in top)
            rows.append({
                "section_id": sid, "rule": "Unusual pattern (ML)", "category": "ML",
                "parameter": ", ".join(FEATURE_LABELS[f] for f in top), "severity": "Low", "priority": 1,
                "start": window["timestamp"].iloc[0], "end": window["timestamp"].iloc[-1],
                "duration_min": (window["timestamp"].iloc[-1] - window["timestamp"].iloc[0]).total_seconds() / 60,
                "peak_value": float(peak["score"]), "threshold": result.threshold,
                "active": window["timestamp"].iloc[-1] == last_ts,
                "explanation": (f"Anomaly score {peak['score']:.3f} exceeded the learned threshold "
                                f"{result.threshold:.3f}. Largest departures from the baseline: {signals}. "
                                "Experimental and approximate; confirm with the rules and field checks."),
                "detector": ML_DETECTOR,
            })
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    out = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
    out.insert(0, "event_id", [f"ML-{i + 1:03d}" for i in range(len(out))])
    return out[EVENT_COLUMNS]
