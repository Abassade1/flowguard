"""Compare detectors against the simulation's own ground truth.

Because the simulation knows exactly where and when it injected an event, we can
measure, for THIS synthetic data only:
* whether a detector raised any alert on an affected section after the event began,
* how long after the event began it did so (detection delay),
* how many alerts it raised that do not correspond to the injected event (false alarms).

These numbers say nothing about performance on a real pipeline.
"""

import pandas as pd

EVALUATION_NOTE = ("Measured on synthetic data with a known injected event. The ML defaults were tuned on this "
                   "same simulator, so these figures are optimistic even for synthetic data. Real pipelines have "
                   "different noise, faults and operating changes; nothing here indicates real-world performance.")


def _matches(events: pd.DataFrame, truth: pd.DataFrame) -> pd.Series:
    """True for events that overlap the injected event on an affected section (starting after it began)."""
    if events.empty or truth.empty:
        return pd.Series(False, index=events.index)
    t = truth.set_index("section_id")
    hit = []
    for _, e in events.iterrows():
        if e["section_id"] not in t.index:
            hit.append(False)
            continue
        w = t.loc[e["section_id"]]
        hit.append(bool(e["start"] >= w["start"] and e["start"] <= w["end"]))
    return pd.Series(hit, index=events.index)


def evaluate(events: pd.DataFrame, truth: pd.DataFrame) -> dict:
    match = _matches(events, truth)
    result = {"alerts": len(events), "false_alarms": int((~match).sum())}
    if truth.empty:
        result.update({"detected": None, "delay_hours": None})
    else:
        detected = bool(match.any())
        delay = ((events.loc[match, "start"].min() - truth["start"].min()).total_seconds() / 3600
                 if detected else None)
        result.update({"detected": detected, "delay_hours": delay})
    return result


def compare_detectors(rule_events: pd.DataFrame, ml_events: pd.DataFrame | None, truth: pd.DataFrame) -> pd.DataFrame:
    rows = [{"detector": "Rules", **evaluate(rule_events, truth)}]
    if ml_events is not None:
        rows.append({"detector": "ML (experimental)", **evaluate(ml_events, truth)})
        rows.append({"detector": "Either (rules or ML)",
                     **evaluate(pd.concat([rule_events, ml_events], ignore_index=True), truth)})
    return pd.DataFrame(rows)
