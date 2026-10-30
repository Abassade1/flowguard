"""Indigenous & community engagement records (Module E).

Principles built into this module (they shape what it deliberately does NOT do):

* Human participation and community decision-making are central. The tool only
  keeps records; it never scores, weighs or monetises Indigenous rights,
  cultural values or community positions.
* Engagement records are not consent. Nothing here calculates or displays
  "consent", "support" or "agreement".
* No locations. Land-use and traditional-activity records have no coordinate
  fields, and the app never infers territories or cultural sites.
* Use authorised information only. Each record has a sensitivity level and can
  record the permission / terms under which information was shared.
* Restricted records are hidden on screen by default and are always excluded
  from exports. NOTE: this is a single-user, local application without user
  accounts, so these labels are handling rules, not security controls.
  Highly sensitive information should stay in systems governed by the
  community concerned.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from modules.projects import DEFAULT_DB_PATH, connect, init_db

SENSITIVITY_LEVELS = ["Public", "Internal", "Restricted"]
RESTRICTED = "Restricted"

COLUMNS = ["id", "project_id", "record_type", "record_date", "title", "details", "community", "category", "status",
           "response", "owner", "due_date", "linked_route", "linked_record_id", "sensitivity", "authorization",
           "created_at", "updated_at"]


@dataclass(frozen=True)
class RecordType:
    key: str
    label: str  # tab title
    singular: str
    fields: dict[str, str]  # column -> form label, in display order
    categories: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()
    default_sensitivity: str = "Internal"
    link_types: tuple[str, ...] = ()  # record types that `linked_record_id` may point to
    guidance: str = ""
    updatable: tuple[str, ...] = field(default=("status", "response", "owner", "due_date", "details"))


RECORD_TYPES: dict[str, RecordType] = {
    "activity": RecordType(
        "activity", "Engagement activities", "Activity",
        {"record_date": "Date", "category": "Activity type", "community": "Communities / organisations involved",
         "title": "Title", "details": "Summary (as agreed with participants)"},
        categories=("Meeting", "Information session", "Site visit", "Workshop", "Correspondence", "Other"),
        guidance="Record what happened and who took part, in terms participants have agreed to."),
    "concern": RecordType(
        "concern", "Concerns & feedback", "Concern",
        {"record_date": "Date raised", "community": "Raised by (community, organisation or role)",
         "category": "Topic", "title": "Concern (short)", "details": "Description (in their words where possible)",
         "linked_route": "Related route option", "status": "Status", "response": "Project response"},
        categories=("Environment", "Water", "Land use & access", "Traditional activities", "Cultural heritage",
                    "Health & safety", "Economic", "Consultation process", "Other"),
        statuses=("Open", "Under review", "Responded", "Resolved", "Unresolved (disagreement remains)"),
        guidance="A response does not resolve a concern. Only those who raised it can say whether it is addressed."),
    "commitment": RecordType(
        "commitment", "Commitments", "Commitment",
        {"category": "Type", "title": "Commitment", "details": "Details", "community": "Made to",
         "owner": "Responsible (project side)", "due_date": "Due date", "linked_record_id": "Related concern",
         "status": "Status"},
        categories=("Community commitment", "Mitigation commitment"),
        statuses=("Planned", "In progress", "Completed", "Cancelled"), link_types=("concern",),
        guidance="Commitments are promises the project makes; track them until delivered."),
    "action": RecordType(
        "action", "Follow-up actions", "Action",
        {"title": "Action", "details": "Details", "owner": "Responsible", "due_date": "Due date",
         "linked_record_id": "Related record", "status": "Status"},
        statuses=("Planned", "In progress", "Completed", "Cancelled"),
        link_types=("activity", "concern", "commitment")),
    "landuse": RecordType(
        "landuse", "Land use & traditional activities", "Consideration",
        {"community": "Information shared by", "category": "Type", "title": "Consideration (short)",
         "details": "Description (no locations or coordinates)",
         "authorization": "Permission / terms of use (who authorised sharing, and for what)",
         "linked_route": "Related route option"},
        categories=("Traditional activities", "Access", "Land use", "Water use", "Seasonal timing", "Other"),
        default_sensitivity=RESTRICTED,
        guidance="Record only what the community has authorised you to hold. Do not record locations of "
                 "culturally sensitive sites; refer to the community's own process instead."),
    "change": RecordType(
        "change", "Project changes from engagement", "Change",
        {"record_date": "Date", "title": "Change made", "details": "Why (what was heard)",
         "linked_record_id": "Arising from", "linked_route": "Affected route option"},
        link_types=("concern", "activity", "landuse"),
        guidance="Documenting changes shows how engagement influenced the project."),
}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def add_record(project_id: int, record_type: str, data: dict, db_path: Path = DEFAULT_DB_PATH) -> int:
    rtype = RECORD_TYPES[record_type]
    if not str(data.get("title", "")).strip():
        raise ValueError("A title is required.")
    values = {k: data.get(k) for k in COLUMNS if k not in ("id", "project_id", "record_type", "created_at", "updated_at")}
    values["sensitivity"] = values.get("sensitivity") or rtype.default_sensitivity
    if values["sensitivity"] not in SENSITIVITY_LEVELS:
        raise ValueError("Unknown sensitivity level.")
    if rtype.statuses and not values.get("status"):
        values["status"] = rtype.statuses[0]
    for k in ("record_date", "due_date"):
        if isinstance(values.get(k), date):
            values[k] = values[k].isoformat()
    init_db(db_path)
    cols = ["project_id", "record_type", *values.keys(), "created_at", "updated_at"]
    with connect(db_path) as conn:
        cur = conn.execute(
            f"INSERT INTO engagement_records ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            [project_id, record_type, *values.values(), _now(), _now()],
        )
        return cur.lastrowid


def update_record(record_id: int, changes: dict, db_path: Path = DEFAULT_DB_PATH) -> None:
    allowed = {k: (v.isoformat() if isinstance(v, date) else v) for k, v in changes.items()
               if k in COLUMNS and k not in ("id", "project_id", "record_type", "created_at")}
    if not allowed:
        return
    allowed["updated_at"] = _now()
    with connect(db_path) as conn:
        conn.execute(f"UPDATE engagement_records SET {', '.join(f'{k} = ?' for k in allowed)} WHERE id = ?",
                     [*allowed.values(), record_id])


def delete_record(record_id: int, db_path: Path = DEFAULT_DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.execute("DELETE FROM engagement_records WHERE id = ?", (record_id,))


def list_records(project_id: int | None, db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    if project_id is None:
        return pd.DataFrame(columns=COLUMNS)
    init_db(db_path)
    with connect(db_path) as conn:
        rows = conn.execute("SELECT * FROM engagement_records WHERE project_id = ? ORDER BY id",
                            (project_id,)).fetchall()
    return pd.DataFrame([dict(r) for r in rows], columns=COLUMNS)


# ---------------------------------------------------------------------------
# Views and summaries
# ---------------------------------------------------------------------------
def redact(records: pd.DataFrame, show_restricted: bool) -> pd.DataFrame:
    """Hide the content of Restricted records unless explicitly shown."""
    out = records.copy()
    if show_restricted or out.empty:
        return out
    mask = out["sensitivity"] == RESTRICTED
    out.loc[mask, "title"] = "[Restricted record]"
    for col in ("details", "response", "authorization"):
        out.loc[mask, col] = "[hidden]"
    return out


def export_view(records: pd.DataFrame) -> pd.DataFrame:
    """What may leave the app: Restricted records are always excluded."""
    return records[records["sensitivity"] != RESTRICTED].drop(columns=["project_id"])


def is_overdue(records: pd.DataFrame, today: date | None = None) -> pd.Series:
    today = pd.Timestamp(today or date.today())
    due = pd.to_datetime(records["due_date"], errors="coerce")  # compare as timestamps (works when all are empty)
    open_ = ~records["status"].isin(["Completed", "Cancelled"])
    return due.notna() & (due < today) & open_


def summarise(records: pd.DataFrame, today: date | None = None) -> dict:
    by = records.groupby("record_type").size() if not records.empty else pd.Series(dtype=int)
    concerns = records[records["record_type"] == "concern"]
    trackable = records[records["record_type"].isin(["commitment", "action"])]
    return {
        "activities": int(by.get("activity", 0)),
        "concerns": int(by.get("concern", 0)),
        "open_concerns": int((~concerns["status"].isin(["Resolved"])).sum()),
        "unresolved": int((concerns["status"] == "Unresolved (disagreement remains)").sum()),
        "commitments": int(by.get("commitment", 0)),
        "open_items": int((~trackable["status"].isin(["Completed", "Cancelled"])).sum()),
        "overdue": int(is_overdue(trackable, today).sum()) if not trackable.empty else 0,
        "changes": int(by.get("change", 0)),
        "landuse": int(by.get("landuse", 0)),
    }


def route_social_status(records: pd.DataFrame, strategy_key: str, not_assessed_text: str) -> str:
    """Informational text for a route option (never a score)."""
    if records.empty or not (records["record_type"] == "activity").any():
        return not_assessed_text
    linked = records[(records["linked_route"] == strategy_key)]
    open_concerns = linked[(linked["record_type"] == "concern") & (linked["status"] != "Resolved")]
    considerations = linked[linked["record_type"] == "landuse"]
    parts = []
    if len(open_concerns):
        parts.append(f"{len(open_concerns)} open concern(s) linked")
    if len(considerations):
        parts.append(f"{len(considerations)} land-use/traditional-activity consideration(s) linked")
    if not parts:
        return ("Engagement under way; nothing linked to this option yet "
                "(not evidence of no concern). Not scored.")
    return "; ".join(parts) + ". See engagement records. Not scored."
