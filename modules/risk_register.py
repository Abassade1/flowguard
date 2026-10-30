"""Risk & mitigation register.

Risks are rated by people on a 5 x 5 likelihood x consequence matrix, before
(inherent) and after (residual) mitigation. The app can SUGGEST risks it can see
in the project data (hard constraints, screening flags, data gaps, cost exposure,
unresolved concerns), but a suggestion only enters the register when a person
reviews it and sets the ratings. Nothing is added or scored automatically.

Social and Indigenous matters appear as obligations to address through engagement,
never as costs or penalties that could be offset by savings elsewhere.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from modules.projects import DEFAULT_DB_PATH, connect, init_db

CATEGORIES = ["Engineering", "Environmental", "Social & community", "Cost & schedule", "Regulatory & permitting",
              "Operational", "Data & uncertainty"]
RISK_STATUSES = ["Open", "Monitoring", "Closed"]
MITIGATION_STATUSES = ["Not started", "In progress", "Implemented"]
SCALE = {1: "Rare / Minor", 2: "Unlikely / Moderate", 3: "Possible / Significant", 4: "Likely / Major",
         5: "Almost certain / Severe"}
LIKELIHOOD_LABELS = {1: "Rare", 2: "Unlikely", 3: "Possible", 4: "Likely", 5: "Almost certain"}
CONSEQUENCE_LABELS = {1: "Minor", 2: "Moderate", 3: "Significant", 4: "Major", 5: "Severe"}
LEVELS = ["Low", "Medium", "High", "Very high"]
LEVEL_COLORS = {"Low": "#2E9E6B", "Medium": "#E0A030", "High": "#D64545", "Very high": "#8B1A1A"}

COLUMNS = ["id", "project_id", "title", "description", "category", "linked_route", "likelihood", "consequence",
           "owner", "mitigation", "mitigation_status", "residual_likelihood", "residual_consequence", "status",
           "source", "created_at", "updated_at"]


def rating(likelihood: int, consequence: int) -> int:
    return int(likelihood) * int(consequence)


def level(score: int) -> str:
    if score >= 15:
        return "Very high"
    if score >= 10:
        return "High"
    if score >= 5:
        return "Medium"
    return "Low"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _check(data: dict) -> None:
    if not str(data.get("title", "")).strip():
        raise ValueError("A risk title is required.")
    for k in ("likelihood", "consequence", "residual_likelihood", "residual_consequence"):
        if k in data and data[k] is not None and int(data[k]) not in range(1, 6):
            raise ValueError("Ratings must be between 1 and 5.")
    if data.get("category", CATEGORIES[0]) not in CATEGORIES:
        raise ValueError("Unknown category.")


def add_risk(project_id: int, data: dict, db_path: Path = DEFAULT_DB_PATH) -> int:
    _check(data)
    init_db(db_path)
    values = {k: data.get(k) for k in COLUMNS if k not in ("id", "project_id", "created_at", "updated_at")}
    values["status"] = values.get("status") or "Open"
    values["mitigation_status"] = values.get("mitigation_status") or "Not started"
    values["source"] = values.get("source") or "User"
    cols = ["project_id", *values, "created_at", "updated_at"]
    with connect(db_path) as conn:
        cur = conn.execute(f"INSERT INTO risks ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                           [project_id, *values.values(), _now(), _now()])
        return cur.lastrowid


def update_risk(risk_id: int, changes: dict, db_path: Path = DEFAULT_DB_PATH) -> None:
    _check({"title": "x", **changes})
    allowed = {k: v for k, v in changes.items() if k in COLUMNS and k not in ("id", "project_id", "created_at")}
    if not allowed:
        return
    allowed["updated_at"] = _now()
    with connect(db_path) as conn:
        conn.execute(f"UPDATE risks SET {', '.join(f'{k} = ?' for k in allowed)} WHERE id = ?",
                     [*allowed.values(), risk_id])


def delete_risk(risk_id: int, db_path: Path = DEFAULT_DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.execute("DELETE FROM risks WHERE id = ?", (risk_id,))


def list_risks(project_id: int | None, db_path: Path = DEFAULT_DB_PATH) -> pd.DataFrame:
    """All risks with computed inherent/residual scores and levels."""
    if project_id is None:
        df = pd.DataFrame(columns=COLUMNS)
    else:
        init_db(db_path)
        with connect(db_path) as conn:
            rows = conn.execute("SELECT * FROM risks WHERE project_id = ? ORDER BY id", (project_id,)).fetchall()
        df = pd.DataFrame([dict(r) for r in rows], columns=COLUMNS)
    df["score"] = [rating(l, c) for l, c in zip(df["likelihood"], df["consequence"])]
    df["level"] = df["score"].map(level)
    # Without residual ratings, the residual rating equals the inherent one. (NaN is truthy, so test explicitly.)
    df["residual_score"] = [rating(l if pd.notna(l) else lk, c if pd.notna(c) else ck) for l, c, lk, ck in
                            zip(df["residual_likelihood"], df["residual_consequence"], df["likelihood"],
                                df["consequence"])]
    df["residual_level"] = df["residual_score"].map(level)
    return df


def matrix_counts(risks: pd.DataFrame, residual: bool = False) -> pd.DataFrame:
    """5 x 5 table of risk counts (rows: likelihood 5 -> 1, columns: consequence 1 -> 5), open risks only."""
    open_risks = risks[risks["status"] != "Closed"]
    lk = open_risks["residual_likelihood"].fillna(open_risks["likelihood"]) if residual else open_risks["likelihood"]
    ck = open_risks["residual_consequence"].fillna(open_risks["consequence"]) if residual else open_risks["consequence"]
    grid = pd.DataFrame(0, index=range(5, 0, -1), columns=range(1, 6))
    for l, c in zip(lk.astype(int), ck.astype(int)):
        grid.loc[l, c] += 1
    return grid


# ---------------------------------------------------------------------------
# Suggestions from project data (reviewed by people before entering the register)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Suggestion:
    key: str  # stable identifier so accepted/dismissed suggestions are not offered again
    title: str
    category: str
    description: str
    linked_route: str | None = None
    source: str = ""


def suggest_risks(routes: pd.DataFrame, screening: pd.DataFrame, design_strategy: str | None,
                  engagement_summary: dict, budget_cad: float, capex_cad_by_strategy: dict[str, float],
                  top_cost_driver: str | None, data_register: list[dict]) -> list[Suggestion]:
    """Risks visible in the project data. The modelled route gets route-specific suggestions."""
    out: list[Suggestion] = []
    names = routes.set_index("strategy")["name"].to_dict()

    for _, r in routes[routes["strategy"] != "REF"].iterrows():
        if r["hard_constraints"]:
            out.append(Suggestion(
                f"hard:{r['strategy']}", f"Hard constraint on {r['name']}", "Regulatory & permitting",
                f"{r['hard_constraints']}. Must be avoided or formally resolved with the responsible authority.",
                r["strategy"], "Route Planning"))

    if design_strategy in names:
        rid = routes.set_index("strategy").loc[design_strategy, "route_id"]
        flagged = screening[(screening["route_id"] == rid) & (screening["level"] == "High")]
        for _, s in flagged.iterrows():
            out.append(Suggestion(
                f"env:{design_strategy}:{s['theme']}", f"{s['label']}: high screening flag ({names[design_strategy]})",
                "Environmental", f"{s['explanation']} Screening only; requires professional assessment.",
                design_strategy, "Environmental Intelligence"))
        capex = capex_cad_by_strategy.get(design_strategy)
        if budget_cad and capex and capex > budget_cad:
            out.append(Suggestion(
                f"budget:{design_strategy}", f"Estimated CAPEX exceeds budget ({names[design_strategy]})",
                "Cost & schedule", f"Illustrative CAPEX is {capex / budget_cad:.0%} of the proposed budget.",
                design_strategy, "Cost Optimizer"))

    if top_cost_driver:
        out.append(Suggestion(
            f"driver:{top_cost_driver}", f"Cost uncertainty: {top_cost_driver}", "Cost & schedule",
            f"'{top_cost_driver}' moves CAPEX most in the sensitivity analysis; it should be firmed up first.",
            None, "Cost Optimizer"))

    for t in screening.loc[screening["level"] == "Not assessed", "label"].unique():
        out.append(Suggestion(f"gap:{t}", f"Data gap: {t}", "Data & uncertainty",
                              f"{t} has not been assessed because data is missing. Impacts are unknown, not absent.",
                              None, "Environmental Intelligence"))
    synthetic = [r["category"] for r in data_register if r["data_type"] == "Synthetic demonstration"]
    if synthetic:
        out.append(Suggestion("gap:synthetic", "Decisions rest on synthetic demonstration data", "Data & uncertainty",
                              "Synthetic (not real) data is used for: " + ", ".join(synthetic) + ".",
                              None, "Data-source register"))

    if engagement_summary.get("activities", 0) == 0:
        out.append(Suggestion(
            "social:none", "No engagement recorded yet", "Social & community",
            "No engagement with rights-holders or communities has been recorded. Indigenous rights and community "
            "considerations cannot be understood without it, and they cannot be offset by cost or schedule savings.",
            None, "Indigenous & Community Engagement"))
    if engagement_summary.get("open_concerns", 0):
        out.append(Suggestion(
            "social:open", f"{engagement_summary['open_concerns']} community concern(s) not resolved",
            "Social & community",
            "Concerns raised through engagement remain open. They must be addressed with the people who raised "
            "them; recording them is not resolution, and engagement is not consent.",
            None, "Indigenous & Community Engagement"))
    if engagement_summary.get("overdue", 0):
        out.append(Suggestion(
            "social:overdue", f"{engagement_summary['overdue']} commitment(s) or action(s) overdue", "Social & community",
            "Overdue commitments can damage trust and the project's licence to operate.",
            None, "Indigenous & Community Engagement"))
    return out


def suggestion_source(s: Suggestion) -> str:
    """Stored in the risk's `source` so accepted suggestions are not offered again."""
    return f"Suggested by {s.source} [{s.key}]"


def accepted_keys(risks: pd.DataFrame) -> set[str]:
    keys = set()
    for src in risks["source"].dropna():
        if "[" in src and src.endswith("]"):
            keys.add(src[src.rindex("[") + 1:-1])
    return keys
