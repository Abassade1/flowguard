"""Investor decision-support reports (PDF and CSV bundle).

`build_report` gathers results from every module into a `Report`: plain sections
with paragraphs, bullet points and tables. Every section states which DATA CLASSES
its content belongs to, so readers can tell verified data from user inputs,
assumptions, estimates and synthetic demonstration data. `to_pdf` and `to_csv_zip`
then render the same Report.

Never included: Restricted engagement records (only counts and exportable records).
"""

import io
import zipfile
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from fpdf import FPDF
from fpdf.fonts import FontFace

from modules.config import (
    APP_NAME, APP_VERSION, ESTIMATE_DISCLAIMER, MONITORING_DISCLAIMER, SOCIAL_NOT_ASSESSED,
)
from modules.cost_engine import COST_GROUPS
from modules.environmental_engine import ASSESSMENT_DISCLAIMER, level_matrix
from modules.projects import Project, fmt_money, to_currency
from modules.risk_register import CONSEQUENCE_LABELS, LEVEL_COLORS, LIKELIHOOD_LABELS, level, matrix_counts
from modules.route_engine import explain_tradeoffs

DATA_CLASSES = {
    "VERIFIED": "Confirmed against a cited source (marked Verified in the project's assumptions register).",
    "USER INPUT": "Entered or uploaded by the project team; not independently verified by this tool.",
    "ASSUMPTION": "A stated planning assumption or an illustrative default value.",
    "ESTIMATE": "Calculated by this tool from inputs and assumptions; inherits their uncertainty.",
    "SYNTHETIC": "Generated demonstration data. Not real; no decision should rest on it.",
}
STATUS_TO_CLASS = {"Verified": "VERIFIED", "Assumed": "ASSUMPTION", "Estimate": "ESTIMATE", "Synthetic": "SYNTHETIC"}

REPORT_STATUS = ("Decision-support draft. All figures are illustrative until validated with project-specific "
                 "engineering, environmental, commercial and community information. This report does not select "
                 "a route, is not an environmental assessment, and does not represent consent or support from "
                 "any community or rights-holder.")


@dataclass
class Table:
    name: str  # used as the CSV file name
    caption: str
    df: pd.DataFrame


@dataclass
class Section:
    title: str
    classes: list[str]
    paragraphs: list[str] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    chart: dict | None = None  # {"type": "bars" | "risk_matrix", ...}


@dataclass
class Report:
    title: str
    project_name: str
    generated_at: str
    sections: list[Section]


def _sources(register: list[dict]) -> list[str]:
    kinds = {r["data_type"] for r in register}
    out = []
    if "Synthetic demonstration" in kinds:
        out.append("SYNTHETIC")
    if kinds & {"User-supplied", "User-supplied DEM"}:
        out.append("USER INPUT")
    return out


def build_report(project: Project, routes: pd.DataFrame, design_route_id: str, estimates: dict,
                 cost_assumptions: pd.DataFrame, sensitivity: pd.DataFrame, screening: pd.DataFrame,
                 register: list[dict], engagement: dict, commitments: pd.DataFrame, risks: pd.DataFrame) -> Report:
    cur, fx = project.currency, to_currency(1.0, project)
    names = routes.set_index("route_id")["name"].to_dict()
    layer_classes = _sources(register)
    proposed = routes[routes["route_id"] != "REF"]
    sections = []

    # 1. Status & provenance -----------------------------------------------------------
    legend = pd.DataFrame({"data_class": list(DATA_CLASSES), "meaning": list(DATA_CLASSES.values())})
    sources = pd.DataFrame(register)[["category", "layer", "source", "data_type", "confidence"]]
    sources["data_class"] = sources["data_type"].map(
        lambda t: "SYNTHETIC" if t == "Synthetic demonstration" else ("USER INPUT" if t.startswith("User") else "-"))
    sections.append(Section(
        "1. Status of this report and data provenance", ["USER INPUT", "SYNTHETIC"],
        [REPORT_STATUS, "Each section below is tagged with the classes of data it contains."],
        tables=[Table("data_classes", "Data classes used in this report", legend),
                Table("data_sources", "Data sources used for mapping and terrain", sources)]))

    # 2. Project overview ---------------------------------------------------------------
    overview = pd.DataFrame([
        ("Project", project.name), ("Pipeline type", project.pipeline_type),
        ("Start", f"{project.start_name} ({project.start_lat:.4f}, {project.start_lon:.4f})"),
        ("Destination", f"{project.end_name} ({project.end_lat:.4f}, {project.end_lon:.4f})"),
        ("Straight-line distance", f"{project.straight_line_km:,.1f} km"),
        ("Diameter", f"{project.diameter_mm:,.0f} mm"), ("Capacity", f"{project.capacity:,.0f} {project.capacity_unit}"),
        ("Proposed construction budget", fmt_money(project.budget, cur) if project.budget else "not set"),
        ("Route option modelled by the team", names.get(design_route_id, "-")),
    ], columns=["item", "value"]).assign(data_class="USER INPUT")
    assumptions = pd.DataFrame(project.assumptions or [], columns=["assumption", "status", "source"])
    assumptions["data_class"] = assumptions["status"].map(STATUS_TO_CLASS).fillna("ASSUMPTION")
    edited = cost_assumptions[cost_assumptions["status"] == "User-edited"]
    sections.append(Section(
        "2. Project overview and assumptions", ["USER INPUT", "VERIFIED", "ASSUMPTION"],
        [project.description] if project.description else [],
        bullets=[f"{len(edited)} of {len(cost_assumptions)} cost assumptions were edited by the team; the rest are "
                 "illustrative defaults."],
        tables=[Table("project", "Project definition", overview),
                Table("assumptions", "Planning assumptions register", assumptions)]))

    # 3. Routes ---------------------------------------------------------------------------
    rt = routes[["name", "comparison_status", "rank", "score", "length_km", "max_slope_pct", "water_crossings",
                 "wetland_km", "forest_clearing_km", "protected_km", "shared_corridor_km"]].copy()
    rt["rank"] = rt["rank"].map(lambda r: "-" if pd.isna(r) else int(r))
    rt = rt.rename(columns={"comparison_status": "status", "max_slope_pct": "max_grade_pct"})
    rt["data_class"] = "ESTIMATE"
    notes = explain_tradeoffs(routes)
    bullets = [f"{names[rid]}: " + " ".join(line.split(" ", 1)[1] for line in lines[:3])
               for rid, lines in notes.items()]
    sections.append(Section(
        "3. Route alternatives and comparison", ["ESTIMATE", *layer_classes],
        ["Corridors are generated as least-cost paths for several strategies and measured against the mapping "
         "layers. Scores are relative to the alternatives shown and depend on the team's weights. The tool does "
         "not choose a route; hard constraints are never offset by cost. Social and Indigenous considerations "
         "are not scored."],
        bullets=bullets, tables=[Table("routes", "Route alternatives (lengths in km)", rt)]))

    # 4. Costs -------------------------------------------------------------------------------
    costs = pd.DataFrame([{
        "route": names[rid], f"capex_{cur}_M": e.capex * fx / 1e6, f"capex_per_km_{cur}_M": e.capex_per_km * fx / 1e6,
        f"opex_per_year_{cur}_M": e.opex_annual * fx / 1e6, f"lifecycle_pv_{cur}_M": e.lifecycle_pv * fx / 1e6,
        "capex_vs_budget": f"{e.capex * fx / project.budget:.0%}" if project.budget else "-",
    } for rid, e in estimates.items() if rid != "REF"]).assign(data_class="ESTIMATE")
    design = estimates.get(design_route_id)
    tables = [Table("costs", f"Cost summary by route ({cur} millions)", costs)]
    chart = {"type": "bars", "title": f"Estimated CAPEX by route ({cur} M)",
             "labels": list(costs["route"]), "values": list(costs[f"capex_{cur}_M"])}
    if design is not None:
        groups = design.by_group().reindex(COST_GROUPS)
        tables.append(Table("cost_breakdown", f"CAPEX breakdown: {names[design_route_id]} ({cur} millions)",
                            pd.DataFrame({"cost_group": groups.index, f"{cur}_M": groups.to_numpy() * fx / 1e6,
                                          "share": [f"{v / design.capex:.0%}" for v in groups]}).assign(
                                data_class="ESTIMATE")))
    top = sensitivity.head(5)
    if not top.empty:
        tables.append(Table("sensitivity", "Most influential cost assumptions (CAPEX change for +/-25%)",
                            pd.DataFrame({"assumption": top["assumption"],
                                          f"swing_{cur}_M": top["swing"] * fx / 1e6}).assign(data_class="ASSUMPTION")))
    sections.append(Section(
        "4. Costs: CAPEX, OPEX and lifecycle", ["ESTIMATE", "ASSUMPTION"],
        [ESTIMATE_DISCLAIMER] + ([f"Converted from CAD at the team's rate 1 CAD = {project.fx_rate_from_cad} {cur}."]
                                 if cur != "CAD" else []),
        bullets=design.notes[1:] if design is not None else [], tables=tables, chart=chart))

    # 5. Environmental ----------------------------------------------------------------------
    matrix = level_matrix(screening[screening["route_id"] != "REF"]).reset_index().rename(columns={"label": "theme"})
    gaps = sorted(screening.loc[screening["level"] == "Not assessed", "label"].unique())
    synthetic_only = sorted(screening.loc[screening["confidence"].str.startswith("None"), "label"].unique())
    sections.append(Section(
        "5. Environmental considerations (screening)", ["ESTIMATE", *layer_classes],
        [ASSESSMENT_DISCLAIMER],
        bullets=([f"Not assessed (data gaps): {', '.join(gaps)}."] if gaps else [])
        + ([f"Based only on synthetic data: {', '.join(synthetic_only)}."] if synthetic_only else []),
        tables=[Table("environmental_screening", "Screening levels by theme and route", matrix)]))

    # 6. Social & community -----------------------------------------------------------------
    summary = pd.DataFrame([("Engagement activities", engagement.get("activities", 0)),
                            ("Concerns raised", engagement.get("concerns", 0)),
                            ("Concerns not resolved", engagement.get("open_concerns", 0)),
                            ("Unresolved disagreements", engagement.get("unresolved", 0)),
                            ("Open commitments and actions", engagement.get("open_items", 0)),
                            ("Overdue commitments and actions", engagement.get("overdue", 0)),
                            ("Project changes from engagement", engagement.get("changes", 0))],
                           columns=["measure", "count"]).assign(data_class="USER INPUT")
    route_social = proposed[["name", "social_indigenous"]].rename(columns={"social_indigenous": "engagement_status"})
    sections.append(Section(
        "6. Social, Indigenous and community considerations", ["USER INPUT"],
        ["Engagement records are not consent, and nothing in this report measures consent, support or agreement. "
         "Indigenous rights and community considerations are not scored or given a monetary value. Restricted "
         "records are excluded from this report."]
        + ([SOCIAL_NOT_ASSESSED + "."] if engagement.get("activities", 0) == 0 else []),
        tables=[Table("engagement_summary", "Engagement record summary", summary),
                Table("engagement_by_route", "Engagement information by route option (not scored)",
                      route_social.assign(data_class="USER INPUT"))]))

    # 7. Engineering constraints --------------------------------------------------------------
    eng = proposed[["name", "terrain_difficulty", "max_slope_pct", "water_crossings", "new_access_km",
                    "geohazards"]].rename(columns={"max_slope_pct": "max_grade_pct"}).assign(data_class="ESTIMATE")
    sections.append(Section("7. Engineering constraints", ["ESTIMATE", *layer_classes],
                            ["Screening-level indicators from route geometry and terrain; geotechnical, hydrotechnical "
                             "and design studies are still required."],
                            tables=[Table("engineering", "Engineering indicators by route", eng)]))

    # 8. Risks & mitigation commitments --------------------------------------------------------
    open_risks = risks[risks["status"] != "Closed"].sort_values(["residual_score", "score"], ascending=False)
    rk = open_risks[["title", "category", "level", "residual_level", "owner", "mitigation", "mitigation_status",
                     "status"]].fillna("").assign(data_class="USER INPUT")
    mitig = pd.concat([
        open_risks.loc[open_risks["mitigation"].fillna("") != "", ["mitigation", "mitigation_status", "owner"]]
        .rename(columns={"mitigation": "commitment", "mitigation_status": "status"}).assign(origin="Risk register"),
        commitments.rename(columns={"title": "commitment", "owner": "owner"})[["commitment", "status", "owner"]]
        .assign(origin="Engagement commitment") if not commitments.empty else None,
    ], ignore_index=True).fillna("").assign(data_class="USER INPUT")
    grid = matrix_counts(risks, residual=True)
    sections.append(Section(
        "8. Risk register and mitigation commitments", ["USER INPUT"],
        ["Risks are rated by the project team on a 5 x 5 likelihood x consequence scale (inherent = before "
         "mitigation, residual = after). Ratings are professional judgement, not calculations."]
        + ([] if len(risks) else ["No risks have been entered in the register yet."]),
        tables=[Table("risks", "Open risks (highest residual first)", rk),
                Table("mitigation_commitments", "Mitigation commitments", mitig)],
        chart={"type": "risk_matrix", "title": "Open risks by residual rating", "grid": grid}))

    # 9. Data gaps & uncertainty ------------------------------------------------------------------
    gap_bullets = [f"Environmental themes not assessed: {', '.join(gaps)}." if gaps else None,
                   ("Mapping layers and/or terrain include SYNTHETIC demonstration data." if "SYNTHETIC" in layer_classes
                    else None),
                   f"Cultural heritage: not assessed (authorised information required).",
                   (f"Largest cost uncertainty: {top['assumption'].iloc[0]}." if not top.empty else None),
                   "Excluded from costs: stations, terminals, energy (unless entered), financing, tax, inflation.",
                   MONITORING_DISCLAIMER]
    sections.append(Section("9. Data gaps and uncertainty", ["ESTIMATE", "ASSUMPTION", *layer_classes],
                            bullets=[b for b in gap_bullets if b]))

    return Report(f"{APP_NAME}: investor decision-support report", project.name,
                  datetime.now().strftime("%Y-%m-%d %H:%M"), sections)


# ---------------------------------------------------------------------------
# PDF rendering (fpdf2, built-in fonts)
# ---------------------------------------------------------------------------
_REPLACE = {"−": "-", "–": "-", "—": "-", "≤": "<=", "≥": ">=", "→": "->", "←": "<-", "’": "'", "‘": "'",
            "“": '"', "”": '"', "…": "...", "•": "-", "✅": "", "⚠️": "", "⛔": "", "🪨": "", "❓": "", "🤝": "",
            " ": " "}


def _latin1(text) -> str:
    """Built-in PDF fonts only cover Latin-1: swap common symbols and drop anything else."""
    s = str(text)
    for k, v in _REPLACE.items():
        s = s.replace(k, v)
    return s.encode("latin-1", "ignore").decode("latin-1").strip()


def _cell(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.1f}"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return _latin1(v)


class _PDF(FPDF):
    def __init__(self, report: Report):
        super().__init__(orientation="L", format="A4")
        self.report = report
        self.set_auto_page_break(True, margin=14)
        self.set_margins(14, 14, 14)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(110)
        self.cell(0, 6, _latin1(f"{self.report.project_name} - decision-support draft (illustrative)"), align="L")
        self.ln(8)
        self.set_text_color(0)

    def footer(self):
        self.set_y(-10)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(110)
        self.cell(0, 6, _latin1(f"{APP_NAME} {APP_VERSION} - generated {self.report.generated_at} - page {self.page_no()}"),
                  align="C")
        self.set_text_color(0)


def _table(pdf: _PDF, table: Table) -> None:
    df = table.df
    if df.empty:
        pdf.set_font("Helvetica", "I", 9)
        pdf.multi_cell(0, 5, _latin1(f"{table.caption}: no entries."), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        return
    pdf.set_font("Helvetica", "B", 9)
    pdf.multi_cell(0, 5, _latin1(table.caption), new_x="LMARGIN", new_y="NEXT")
    rows = [[_cell(v) for v in r] for r in df.itertuples(index=False)]
    headers = [_latin1(c.replace("_", " ")) for c in df.columns]
    # Width ~ content length, but never narrower than the longest single word (no mid-word breaks).
    def longest_word(text: str) -> int:
        return max((len(w) for w in text.split()), default=1)

    widths = []
    for i, h in enumerate(headers):
        content = min(max(len(r[i]) for r in rows), 60)
        words = max([longest_word(h)] + [longest_word(r[i]) for r in rows])
        widths.append(max(content, words, 5) + 1)
    pdf.set_font("Helvetica", size=7.5)
    pdf.set_fill_color(255, 255, 255)  # fpdf keeps the last fill colour; tables must not inherit it
    with pdf.table(col_widths=widths, text_align="LEFT", line_height=4.2, padding=1,
                   headings_style=FontFace(emphasis="BOLD", fill_color=(235, 238, 242))) as t:
        head = t.row()
        for h in headers:
            head.cell(h)
        for r in rows:
            row = t.row()
            for v in r:
                row.cell(v)
    pdf.ln(3)


def _bars(pdf: _PDF, chart: dict) -> None:
    labels, values = chart["labels"], chart["values"]
    if not values:
        return
    pdf.set_font("Helvetica", "B", 9)
    pdf.multi_cell(0, 5, _latin1(chart["title"]), new_x="LMARGIN", new_y="NEXT")
    x0, label_w, bar_w, h = pdf.l_margin, 70, 150, 6
    vmax = max(values) or 1
    for label, v in zip(labels, values):
        y = pdf.get_y()
        pdf.set_font("Helvetica", size=8)
        pdf.set_xy(x0, y)
        pdf.cell(label_w, h, _latin1(label))
        pdf.set_fill_color(42, 120, 214)
        pdf.rect(x0 + label_w, y + 1, bar_w * v / vmax, h - 2, style="F")
        pdf.set_xy(x0 + label_w + bar_w * v / vmax + 2, y)
        pdf.cell(30, h, f"{v:,.0f}")
        pdf.ln(h)
    pdf.ln(3)


def _risk_matrix(pdf: _PDF, chart: dict) -> None:
    grid: pd.DataFrame = chart["grid"]
    pdf.set_font("Helvetica", "B", 9)
    pdf.multi_cell(0, 5, _latin1(chart["title"]), new_x="LMARGIN", new_y="NEXT")
    size, x0 = 15, pdf.l_margin + 32
    y0 = pdf.get_y() + 2
    if y0 + 6 * size + 12 > pdf.h - pdf.b_margin:
        pdf.add_page()
        y0 = pdf.get_y() + 2
    for i, lk in enumerate(grid.index):
        pdf.set_xy(pdf.l_margin, y0 + i * size)
        pdf.set_font("Helvetica", size=7)
        pdf.cell(32, size, _latin1(f"{lk} {LIKELIHOOD_LABELS[lk]}"))
        for j, ck in enumerate(grid.columns):
            color = LEVEL_COLORS[level(lk * ck)].lstrip("#")
            pdf.set_fill_color(*(int(color[k:k + 2], 16) for k in (0, 2, 4)))
            pdf.rect(x0 + j * size, y0 + i * size, size, size, style="DF")
            n = int(grid.loc[lk, ck])
            if n:
                pdf.set_text_color(255)
                pdf.set_xy(x0 + j * size, y0 + i * size)
                pdf.set_font("Helvetica", "B", 9)
                pdf.cell(size, size, str(n), align="C")
                pdf.set_text_color(0)
    pdf.set_font("Helvetica", size=6.5)
    for j, ck in enumerate(grid.columns):
        pdf.set_xy(x0 + j * size, y0 + 5 * size + 1)
        pdf.cell(size, 4, _latin1(CONSEQUENCE_LABELS[ck]), align="C")
    pdf.set_xy(pdf.l_margin, y0 + 5 * size + 6)
    pdf.set_font("Helvetica", "I", 7)
    pdf.cell(0, 4, "Rows: likelihood. Columns: consequence. Colour shows the rating level (Low / Medium / High / Very high).")
    pdf.ln(8)


def to_pdf(report: Report) -> bytes:
    pdf = _PDF(report)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 20)
    pdf.ln(20)
    pdf.multi_cell(0, 10, _latin1(report.title), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 14)
    pdf.multi_cell(0, 8, _latin1(report.project_name), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(4)
    pdf.multi_cell(0, 6, _latin1(f"Generated {report.generated_at} with {APP_NAME} {APP_VERSION}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)
    pdf.set_fill_color(253, 246, 227)
    pdf.set_font("Helvetica", "B", 10)
    pdf.multi_cell(0, 6, _latin1(REPORT_STATUS), fill=True, border=1, padding=3, new_x="LMARGIN", new_y="NEXT")

    for section in report.sections:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 14)
        pdf.multi_cell(0, 8, _latin1(section.title), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(90)
        pdf.multi_cell(0, 5, _latin1("Data classes: " + ", ".join(dict.fromkeys(section.classes))), new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0)
        pdf.ln(1)
        pdf.set_font("Helvetica", size=9.5)
        for p in section.paragraphs:
            pdf.multi_cell(0, 5, _latin1(p), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1.5)
        for b in section.bullets:
            pdf.set_x(pdf.l_margin + 3)
            pdf.multi_cell(0, 5, _latin1(f"- {b}"), new_x="LMARGIN", new_y="NEXT")
        if section.bullets:
            pdf.ln(2)
        if section.chart and section.chart["type"] == "bars":
            _bars(pdf, section.chart)
        for t in section.tables:
            _table(pdf, t)
        if section.chart and section.chart["type"] == "risk_matrix":
            _risk_matrix(pdf, section.chart)
    return bytes(pdf.output())


# ---------------------------------------------------------------------------
# CSV bundle
# ---------------------------------------------------------------------------
def to_csv_zip(report: Report) -> bytes:
    """A zip of every report table as CSV, plus a README explaining data classes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        readme = [report.title, report.project_name, f"Generated {report.generated_at}", "", REPORT_STATUS, "",
                  "DATA CLASSES (see the data_class column in each file):"]
        readme += [f"  {k}: {v}" for k, v in DATA_CLASSES.items()]
        readme += ["", "FILES:"]
        for s in report.sections:
            for t in s.tables:
                z.writestr(f"{t.name}.csv", t.df.to_csv(index=False))
                readme.append(f"  {t.name}.csv - {t.caption} ({s.title}; classes: {', '.join(dict.fromkeys(s.classes))})")
        z.writestr("README.txt", "\n".join(readme))
    return buf.getvalue()
