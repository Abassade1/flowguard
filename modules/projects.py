"""Pipeline projects: the data model and a small SQLite store.

A `Project` holds everything the user defines on the Project Setup page.
Projects are stored in a local SQLite database file (no server needed).
Each project is saved as one JSON document, so new fields can be added in
later phases without database migrations.

User-supplied GIS layers belong to a project and are stored in a second table
(see `save_layer` / `list_layers`).
"""

import json
import os
import sqlite3
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path

import numpy as np
from pyproj import Geod

ROOT = Path(__file__).resolve().parents[1]
# Set FLOWGUARD_DB to use a different database file (the tests do this).
DEFAULT_DB_PATH = Path(os.environ.get("FLOWGUARD_DB", ROOT / "data" / "database" / "flowguard.db"))
_OLD_DB_PATH = ROOT / "data" / "database" / "pipeline_guardian.db"  # name used before the product became FlowGuard
if "FLOWGUARD_DB" not in os.environ and _OLD_DB_PATH.exists() and not DEFAULT_DB_PATH.exists():
    _OLD_DB_PATH.rename(DEFAULT_DB_PATH)
SAMPLE_PROJECT_PATH = ROOT / "data" / "sample_projects" / "demo_project.json"

ASSUMPTION_STATUSES = ["Verified", "Assumed", "Estimate", "Synthetic"]
CAPACITY_UNITS = ["m³/day", "bbl/day", "10³ m³/day (gas)", "ML/day (water)"]
LENGTH_UNITS = ["km", "mi"]
CURRENCIES = ["CAD", "USD", "EUR"]

_GEOD = Geod(ellps="WGS84")


@dataclass
class Project:
    name: str = "New pipeline project"
    pipeline_type: str = "Oil"
    description: str = ""
    start_name: str = "Start"
    start_lat: float = 53.20
    start_lon: float = -112.90
    end_name: str = "Destination"
    end_lat: float = 52.55
    end_lon: float = -111.40
    approx_length_km: float = 0.0  # user's estimate; 0 = not given
    diameter_mm: float = 610.0
    capacity: float = 0.0
    capacity_unit: str = CAPACITY_UNITS[0]
    budget: float = 0.0  # in `currency`
    currency: str = "CAD"
    fx_rate_from_cad: float = 1.0  # user-supplied: 1 CAD = this many units of `currency`
    length_unit: str = "km"
    assumptions: list[dict] = field(default_factory=list)
    cost_assumptions: dict = field(default_factory=dict)  # only values changed from the defaults
    cost_uncertainty: dict = field(default_factory=dict)  # Monte Carlo ranges changed from the defaults
    dismissed_risk_suggestions: list[str] = field(default_factory=list)  # suggestion keys the user set aside
    design_route_strategy: str = ""  # route option modelled in the twin/monitoring (user's choice)
    construction_progress: dict = field(default_factory=dict)  # section_id -> construction stage
    id: int | None = None
    created_at: str = ""
    updated_at: str = ""

    @property
    def straight_line_km(self) -> float:
        """Geodesic (great-circle) distance between start and destination."""
        _, _, metres = _GEOD.inv(self.start_lon, self.start_lat, self.end_lon, self.end_lat)
        return metres / 1000.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        """Build a Project, ignoring unknown keys (so old saved files still load)."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def validate_project(p: Project) -> list[str]:
    """Return a list of problems. An empty list means the project can be saved."""
    problems = []
    if not p.name.strip():
        problems.append("Project name is required.")
    for label, lat, lon in [("Start", p.start_lat, p.start_lon), ("Destination", p.end_lat, p.end_lon)]:
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            problems.append(f"{label} coordinates are out of range.")
    if not problems:
        d = p.straight_line_km
        if d < 5:
            problems.append("Start and destination must be at least 5 km apart.")
        elif d > 1500:
            problems.append("Start and destination are more than 1,500 km apart; route planning in this "
                            "version supports up to 1,500 km.")
    if p.diameter_mm < 0 or p.capacity < 0 or p.budget < 0:
        problems.append("Diameter, capacity and budget cannot be negative.")
    if p.fx_rate_from_cad <= 0:
        problems.append("Exchange rate must be greater than zero.")
    return problems


def load_sample_project() -> Project:
    data = json.loads(SAMPLE_PROJECT_PATH.read_text(encoding="utf-8"))
    return Project.from_dict(data)


# ---------------------------------------------------------------------------
# SQLite storage
# ---------------------------------------------------------------------------
def connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS layers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                category TEXT NOT NULL,
                name TEXT NOT NULL,
                source TEXT NOT NULL,
                licence TEXT,
                notes TEXT,
                geojson TEXT NOT NULL,
                loaded_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS dems (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                source TEXT NOT NULL,
                licence TEXT,
                notes TEXT,
                file_path TEXT NOT NULL,
                loaded_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS risks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                title TEXT NOT NULL,
                description TEXT,
                category TEXT NOT NULL,
                linked_route TEXT,
                likelihood INTEGER NOT NULL,
                consequence INTEGER NOT NULL,
                owner TEXT,
                mitigation TEXT,
                mitigation_status TEXT,
                residual_likelihood INTEGER,
                residual_consequence INTEGER,
                status TEXT NOT NULL,
                source TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS alert_acks (
                event_key TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                note TEXT,
                actor TEXT,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS engagement_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                record_type TEXT NOT NULL,
                record_date TEXT,
                title TEXT NOT NULL,
                details TEXT,
                community TEXT,
                category TEXT,
                status TEXT,
                response TEXT,
                owner TEXT,
                due_date TEXT,
                linked_route TEXT,
                linked_record_id INTEGER,
                sensitivity TEXT NOT NULL DEFAULT 'Internal',
                authorization TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )


def save_project(project: Project, db_path: Path = DEFAULT_DB_PATH) -> Project:
    """Insert a new project or update an existing one. Returns the saved copy (with id)."""
    init_db(db_path)
    now = datetime.now().isoformat(timespec="seconds")
    saved = Project.from_dict(project.to_dict())
    saved.updated_at = now
    saved.created_at = saved.created_at or now
    with connect(db_path) as conn:
        if saved.id is None:
            cur = conn.execute(
                "INSERT INTO projects (name, data, created_at, updated_at) VALUES (?, '{}', ?, ?)",
                (saved.name, saved.created_at, now),
            )
            saved.id = cur.lastrowid
        conn.execute(
            "UPDATE projects SET name = ?, data = ?, updated_at = ? WHERE id = ?",
            (saved.name, json.dumps(saved.to_dict()), now, saved.id),
        )
    return saved


def load_project(project_id: int, db_path: Path = DEFAULT_DB_PATH) -> Project:
    init_db(db_path)
    with connect(db_path) as conn:
        row = conn.execute("SELECT data FROM projects WHERE id = ?", (project_id,)).fetchone()
    if row is None:
        raise KeyError(f"No project with id {project_id}")
    return Project.from_dict(json.loads(row["data"]))


def list_projects(db_path: Path = DEFAULT_DB_PATH) -> list[dict]:
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute("SELECT id, name, updated_at FROM projects ORDER BY updated_at DESC").fetchall()
    return [dict(r) for r in rows]


def delete_project(project_id: int, db_path: Path = DEFAULT_DB_PATH) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))


def save_layer(project_id: int, category: str, name: str, source: str, geojson: str,
               licence: str = "", notes: str = "", db_path: Path = DEFAULT_DB_PATH) -> int:
    init_db(db_path)
    with connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO layers (project_id, category, name, source, licence, notes, geojson, loaded_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (project_id, category, name, source, licence, notes, geojson,
             datetime.now().isoformat(timespec="seconds")),
        )
        return cur.lastrowid


def list_layers(project_id: int | None, db_path: Path = DEFAULT_DB_PATH) -> list[dict]:
    """All user layers for a project, including their GeoJSON text."""
    if project_id is None:
        return []
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM layers WHERE project_id = ? ORDER BY id", (project_id,)).fetchall()
    return [dict(r) for r in rows]


def delete_layer(layer_id: int, db_path: Path = DEFAULT_DB_PATH) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute("DELETE FROM layers WHERE id = ?", (layer_id,))


# ---------------------------------------------------------------------------
# Units and currency
# ---------------------------------------------------------------------------
KM_PER_MILE = 1.609344


def to_length(km: float, unit: str) -> float:
    return km / KM_PER_MILE if unit == "mi" else km


def to_currency(cad: float | np.ndarray, project: Project):
    """Convert a CAD amount into the project's currency using its user-supplied rate."""
    return cad * (1.0 if project.currency == "CAD" else project.fx_rate_from_cad)


def fmt_money(value: float, currency: str) -> str:
    """Compact money format, e.g. 'CAD 412.3 M'."""
    if value is None or np.isnan(value):
        return "-"
    if abs(value) >= 1e9:
        return f"{currency} {value / 1e9:,.2f} B"
    return f"{currency} {value / 1e6:,.1f} M"
