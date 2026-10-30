# FlowGuard

AI-assisted pipeline planning, investment decision-support, environmental assessment support,
digital twin and operational monitoring for oil, natural gas and water pipelines.

> **Status: all 7 phases complete, plus probabilistic costs (version 1.1).** Projects, route planning, cost engine, environmental
> screening, engagement records, 3D digital twin, monitoring with explainable rules and optional ML,
> risk register and investor reports. Still a decision-support prototype: see *Responsible use* below.

---

## Quick start (Mac)

You need Python 3.11+ (this project was built with Python 3.14).

```bash
cd ~/Flowguide
python3 -m venv .venv               # one-time: create an isolated Python environment
source .venv/bin/activate           # activate it (do this in every new terminal)
pip install -r requirements.txt     # one-time: install the packages
```

### Start the application

```bash
source .venv/bin/activate
streamlit run app.py
```

Your browser opens at **http://localhost:8501**.

### Stop the application

Press **Ctrl + C** in the terminal where it is running.

> Tip: Streamlit reruns a page when you save it. If a change in `modules/` doesn't
> appear, stop the app (Ctrl + C) and start it again.

### Run the tests

```bash
source .venv/bin/activate
pytest
```

---

## Temporary online hosting (free)

GitHub Pages only serves static files, so it cannot run a Streamlit (Python) app. Two free options that work straight from the GitHub repository:

**GitHub Codespaces (temporary, private by default).** On the repository page choose **Code → Codespaces → Create codespace on main**. The `.devcontainer/` config installs the requirements and starts the app; it opens in a preview tab on port 8501. To let others see it, open the **Ports** tab, right-click port 8501 → **Port visibility → Public**, and share the URL. The codespace stops after inactivity (30 minutes by default) and the link stops working; delete it from github.com/codespaces when finished. Free accounts get a monthly allowance of core-hours.

**Streamlit Community Cloud (stays up, public link).** Sign in at share.streamlit.io with GitHub, choose **Create app**, pick this repository, branch `main`, main file `app.py`, and under **Advanced settings** choose Python 3.13. Delete the app from the dashboard when you no longer need it.

On both, the SQLite database is temporary: projects saved online are lost when the codespace or app restarts. The demo data is synthetic, so do not enter confidential or culturally sensitive information on a public link.

---

## What you can do

| Page | What it does |
|---|---|
| **Executive Overview** | Headline metrics, status strip along the pipeline, top alerts, route snapshot |
| **Project Setup** | Create / save / reopen / delete projects (local SQLite). Set start & destination by clicking the map or typing coordinates. Diameter, capacity, budget, currency (CAD default, your own exchange rate), km/mi, assumptions register |
| **Cost Optimizer** | 22 editable cost assumptions (saved with the project), CAPEX / OPEX / lifecycle cost (undiscounted and present value) per route, cost per km, CAPEX vs budget, breakdown by cost group, line-by-line basis of estimate, tornado sensitivity, **Monte Carlo cost ranges** (P10/P50/P80/P90, S-curves, chance of staying within budget, contingency needed for P80 vs assumed; editable ranges saved with the project), cumulative cost over the design life, CSV export |
| **Environmental Intelligence** | 11 screening themes per route (wetlands, water & riparian, protected areas, forest, fragmentation, habitat & biodiversity, erosion / landslide / flood proxies, disturbance, restoration) with level, data status and confidence; data gaps shown as *Not assessed*; features crossed or nearby; mitigation ideas for professional review; CSV export |
| **Indigenous & Community Engagement** | Records of engagement activities, concerns & feedback, commitments, follow-up actions, land-use & traditional-activity considerations (no locations), and project changes; sensitivity levels (Restricted hidden by default, never exported); overdue tracking; route-linked concerns shown on Route Planning as information only |
| **Route Planning** | Upload a GeoTIFF elevation model (DEM) to replace synthetic terrain. Generated corridors on a map (least-cost paths for 4 strategies + straight-line reference), constraint layers, upload your own GIS layers with source records, elevation profiles, multi-criteria comparison, trade-offs, CSV export |
| **3D Digital Twin** | The project's real coordinates on its terrain (uploaded DEM or labelled synthetic). **Planning** mode: constraint layers, all route options, indicative facilities. **Construction** mode: sections coloured by recorded construction stage, length-weighted progress. **Operations** mode: sections coloured by simulated status, sensor readings, time replay. You choose which route option to model; scene export (JSON) for other 3D tools |
| **FlowGuard Monitoring** | 7 scenarios (incl. silent flow-meter fault and sensor drift), explainable rules with a leak-vs-meter cross-check, optional ML anomaly detection (robust covariance or Isolation Forest, advisory only), replay with no look-ahead, operator acknowledgements, event history with filters, detector comparison against the simulation's ground truth, CSV export |
| **Risk & Mitigation Register** | 5 × 5 likelihood × consequence register (inherent and residual), owners, mitigations, status; risk suggestions drawn from every module that you review and rate before they are added; inherent/residual matrices; mitigation commitments (risk register + engagement); CSV export |
| **Reports & Export** | Investor dashboard (route options, CAPEX range, lifecycle cost, budget, risks, environmental and community summaries, data gaps) and downloadable **PDF report** and **CSV bundle**; every section is tagged VERIFIED / USER INPUT / ASSUMPTION / ESTIMATE / SYNTHETIC; Restricted engagement records are never included |

**Sidebar (every page):** choose the pipeline type and a simulation scenario (normal, possible
leak, pressure surge, sensor fault, temperature excursion), the affected section, the onset time and
the severity. Advanced settings: length, number of sections, time window, noise, random seed.

### Try this

1. Open **Project Setup**, click **Load demo**, click the map to move the start point, then **Save project**.
2. Open **Route Planning**. Compare the corridors; hover the map features (all marked SYNTHETIC).
   Turn off *Treat protected areas as impassable* to see what changes.
3. Upload a GeoJSON layer (e.g. a protected-area polygon) under *Add your own GIS layer*. It replaces
   the synthetic layer for that category and is listed in the data-source register.
4. Open **Cost Optimizer**. Compare route costs, pick a route to see its basis of estimate, and check
   the tornado chart. Open *Cost assumptions*, raise the steel price, and watch every figure (and the
   Route Planning scores) update. Scroll to *Probabilistic estimate*: compare P50/P80 with the budget and see
   how much contingency P80 really needs. Save the project to keep your assumptions and ranges.
5. Open **Environmental Intelligence**. Note that *Wildlife habitat & biodiversity* is **Not assessed**
   (no data), and every other theme says its confidence is *None* because the layers are synthetic.
6. Save the project, then open **Indigenous & Community Engagement**. Add an activity, then a concern linked
   to a route option; Route Planning shows it as information (never as a score). Add a land-use
   consideration: it is Restricted by default, hidden on screen and excluded from the CSV export.
7. Open **FlowGuard Monitoring**. With the default "Possible leak" scenario, S04 is in
   **Alert** and the alert explains the flow imbalance that triggered it.
8. Choose the scenario *Flow meter fault (bias)*: the rules flag a possible leak, and the alert's
   cross-check says no downstream pressure drop was seen (a meter fault is possible). Then choose
   *Pressure sensor drift*, open *Experimental machine-learning detection* and enable it: the rules stay
   quiet, the ML flags S04 as Advisory, and *Detector comparison* shows the difference. Acknowledge an
   alert to record what you did about it.
9. Drag **Replay time** back before hour 48: S04 returns to Normal. Or switch on **Live-style replay**.
10. In the sidebar, set **Event severity** to 0.1. The leak is no longer detected. Small leaks can
   be missed, which is an honest limitation of simple rules.
11. Choose **Sensor fault**. S04 shows **Sensor issue** (instrument problem), not a process alarm.
12. Open **Rule thresholds** and change the flow imbalance limit. The alerts update immediately.
13. On **Route Planning**, set every weight to 0 except *Forest clearing*. The ranking changes.
   The straight-line reference stays unranked because it crosses a protected area.
14. Open **Risk & Mitigation Register**: review the suggested risks, rate one and add it. Then open
    **Reports & Export**, look at the investor dashboard, and download the PDF and the CSV bundle.
15. On **3D Digital Twin**, choose the route option to model, then switch between Planning,
    Construction (set a few section stages) and Operations (hover the sensor diamonds). The sidebar's
    simulation now uses that route's length; the Monitoring page follows it.

---

## Project structure

```
Flowguide/
├── app.py                     Entry point: page frame, navigation, shared sidebar
├── requirements.txt           Python packages
├── pytest.ini                 Test settings
├── README.md / TODO.md        Documentation and progress tracking
├── .streamlit/config.toml     Theme and server settings
│
├── app_pages/                 One file per screen (display only, no calculations)
│   ├── 0_Executive_Overview.py
│   ├── 1_Project_Setup.py
│   ├── 2_Route_Planning.py
│   ├── 3_Cost_Optimizer.py
│   ├── 4_Environmental_Intelligence.py
│   ├── 5_Community_Engagement.py
│   ├── 6_Digital_Twin.py
│   ├── 7_Pipeline_Monitoring.py
│   ├── 8_Risk_Register.py
│   └── 9_Investor_Reports.py
│
├── modules/                   Business logic: plain Python, independently testable
│   ├── config.py              Disclaimers, pipeline profiles, colours
│   ├── monitoring_engine.py   Synthetic sensor simulation + scenarios
│   ├── anomaly_detection.py   Transparent rule-based alerts, leak cross-check, no-look-ahead replay
│   ├── ml_detection.py        Optional scikit-learn anomaly detection with approximate explanations
│   ├── detector_evaluation.py Rules vs ML against the simulation's ground truth
│   ├── alert_acks.py          Operator acknowledgements (record-keeping only)
│   ├── projects.py            Project model, SQLite store (projects + uploaded layers), units & currency
│   ├── gis_layers.py          UTM frames, synthetic layers & terrain, reading user GIS files
│   ├── routing.py             Least-cost corridor generation (Dijkstra) + route measurement
│   ├── maps.py                Folium map builders
│   ├── cost_engine.py         CAPEX / OPEX / lifecycle estimates + sensitivity (editable assumptions)
│   ├── cost_uncertainty.py    Monte Carlo cost simulation (P-values, implied contingency)
│   ├── environmental_engine.py  Environmental screening themes, confidence, conflicts, mitigation ideas
│   ├── social_impact.py       Engagement records (SQLite), redaction, export rules, summaries
│   ├── route_engine.py        Multi-criteria scoring + trade-off explanations
│   ├── elevation.py           Elevation sources: synthetic terrain or uploaded GeoTIFF DEM
│   ├── digital_twin.py        Renderer-independent 3D scene + Plotly renderer (3 modes) + JSON export
│   ├── risk_register.py       Risk register (SQLite), ratings, matrices, suggestions
│   ├── reporting.py           Report builder with data-class tags; PDF (fpdf2) and CSV zip export
│   ├── charts.py              Reusable Plotly charts
│   └── ui.py                  Shared Streamlit pieces (sidebar, banners, caching)
│
├── data/
│   ├── sample_projects/       demo_project.json (synthetic demo)
│   ├── database/              flowguard.db (created automatically; your saved projects)
│   └── dem/                   cropped elevation models, one folder per project
├── tests/                     pytest tests (modules + every page)
└── assets/                    Images/logos (empty for now)
```

**Why `app_pages/` instead of `pages/`?** Streamlit treats a folder named exactly `pages/` as
an older, automatic navigation mode, which bypasses `app.py`. Using `app_pages/` with
`st.navigation` keeps one shared sidebar and settings that survive page changes.

**Maps and internet:** the OpenStreetMap basemap needs an internet connection (free; please respect
the OSM tile usage policy). Choose *Basemap: None (offline)* to work without it. Everything else runs
locally, and no project data is sent anywhere.

**Your data:** saved projects live in `data/database/flowguard.db`, and elevation models in
`data/dem/`. Back up both to keep them. Delete them to start fresh.

**Real terrain (optional):** download a free GeoTIFF elevation model that covers both ends of your
route (for example Natural Resources Canada elevation data on open.canada.ca, or Copernicus DEM GLO-30),
then on **Route Planning → Elevation model (DEM)** upload it with its source. It replaces the synthetic
terrain for routing slopes, elevation profiles, environmental proxies and the 3D twin. Areas the file
does not cover have *unknown* elevation; nothing is filled in.

**Design rule:** pages *display*, modules *calculate*. Only `modules/ui.py` imports Streamlit,
so all the calculations can be tested without a browser.

---

## Responsible-use principles built into the design

- Synthetic data is labelled on every page and drawn with dashed outlines on maps. Missing data is never
  presented as "no impact".
- Every GIS layer has a source record. User uploads require a named source; synthetic layers say so.
- Route comparison is multi-criteria and never picks a winner automatically. Hard constraints are
  shown separately and cannot be offset by cost.
- Social, Indigenous and cultural considerations are **not scored or monetised**. Engagement records
  are **not consent**, and nothing in the app measures consent, support or agreement.
- No locations are stored for land use or traditional activities, and territories or cultural sites are
  never inferred. Restricted records are hidden by default and never exported. This is a local,
  single-user app: sensitivity labels are handling rules, not security controls.
- Environmental screening is not an environmental assessment. Missing data is *Not assessed*, never "Low".
  No synthetic wildlife habitat is generated: without real data, habitat stays a visible data gap.
- Alerts explain why they fired. The system gives no automated control. All operational
  decisions stay with qualified people following validated procedures.
- ML is optional and advisory: it never raises a section above *Advisory*, its explanations are labelled
  approximate, and its measured performance is on synthetic data it was tuned on (stated in the app).
- Replay uses only data available at that moment (no look-ahead).
- Estimates are labelled illustrative until validated with project-specific data. Every cost line shows
  its basis, and each assumption is marked "Illustrative default" or "User-edited". Costs the tool cannot
  estimate (e.g. pumping energy) are shown as excluded rather than guessed.

See [TODO.md](TODO.md) for completed and planned features.
