"""Operator acknowledgement of (simulated) alerts.

Acknowledging is a record-keeping step for people: it never changes detection,
never silences rules, and never triggers any control action.

Simulated events are recreated from their settings each run, so an alert is
identified by a stable key: simulation settings + section + rule/detector + start time.
"""

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from modules.projects import DEFAULT_DB_PATH, connect, init_db

ACK_STATUSES = ["New", "Acknowledged", "Under investigation", "Closed: false alarm",
                "Closed: issue confirmed (simulated)"]


def config_key(config) -> str:
    """Short fingerprint of the simulation settings."""
    return hashlib.sha1(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()[:10]


def event_key(cfg_key: str, event: pd.Series) -> str:
    return f"{cfg_key}|{event['section_id']}|{event['detector']}|{event['rule']}|{pd.Timestamp(event['start']).isoformat()}"


def set_ack(key: str, status: str, note: str = "", actor: str = "", db_path: Path = DEFAULT_DB_PATH) -> None:
    if status not in ACK_STATUSES:
        raise ValueError("Unknown acknowledgement status.")
    init_db(db_path)
    with connect(db_path) as conn:
        if status == "New":
            conn.execute("DELETE FROM alert_acks WHERE event_key = ?", (key,))
        else:
            conn.execute(
                "INSERT INTO alert_acks (event_key, status, note, actor, updated_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(event_key) DO UPDATE SET status=excluded.status, note=excluded.note, "
                "actor=excluded.actor, updated_at=excluded.updated_at",
                (key, status, note, actor, datetime.now().isoformat(timespec="seconds")))


def acks_for(cfg_key: str, db_path: Path = DEFAULT_DB_PATH) -> dict[str, dict]:
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM alert_acks WHERE event_key LIKE ?", (f"{cfg_key}|%",)).fetchall()
    return {r["event_key"]: dict(r) for r in rows}


def with_acks(events: pd.DataFrame, cfg_key: str, acks: dict[str, dict]) -> pd.DataFrame:
    """Add ack_key / ack_status / ack_note columns (status 'New' when never acknowledged)."""
    out = events.copy()
    keys = [event_key(cfg_key, e) for _, e in out.iterrows()]
    out["ack_key"] = keys
    out["ack_status"] = [acks.get(k, {}).get("status", "New") for k in keys]
    out["ack_note"] = [acks.get(k, {}).get("note", "") for k in keys]
    return out
