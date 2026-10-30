# FlowGuard: TODO

Legend: [x] done · [ ] planned

## Phase 1: Foundation ✅ (2026-10-30)
- [x] Project structure, virtual environment, requirements, README, tests
- [x] Multipage Streamlit app with grouped navigation and consistent styling
- [x] Shared sidebar: pipeline type, scenario, affected section, onset, severity, advanced settings
- [x] Synthetic sensor simulation (pressure, temperature, inlet/outlet flow, sensor health)
- [x] Scenarios: normal, possible leak (flow imbalance), pressure surge, sensor fault, temperature excursion
- [x] Rule-based alerts with explanations, severity, priority, event grouping
- [x] Editable demonstration thresholds
- [x] Section status (Normal / Advisory / Sensor issue / Watch / Alert)
- [x] Trend charts, section detail chart with thresholds and event shading
- [x] Time replay slider + live-style auto replay
- [x] Event history with filters + CSV export
- [x] 3D digital twin (Plotly): route, sections coloured by status, sensor stations, terrain, selection
- [x] Route comparison: 4 synthetic alternatives, weighted multi-criteria score, hard constraints unranked,
      trade-off heatmap, plain-language trade-offs, CSV export
- [x] Placeholder pages for later modules
- [x] Synthetic-data, monitoring and estimate disclaimers throughout

## Phase 2: Geographic route planning ✅ (2026-10-30)
- [x] Project Setup page (name, type, start/destination, length, diameter, capacity, budget, assumptions)
- [x] Save / reopen / delete projects (SQLite in `data/database/`)
- [x] Units (km / mi) and currency (CAD default; user-entered exchange rate, no paid API)
- [x] Map interface (Folium) with click-to-set start/destination
- [x] Least-cost corridor generation for 4 strategies + straight-line reference (own Dijkstra, no extra library)
- [x] Routes measured in a projected CRS (UTM): length, water crossings, wetland, forest, protected area,
      shared corridor, new-access estimate, grades from the elevation profile
- [x] Protected areas as hard constraints (impassable by default; crossing routes flagged and unranked)
- [x] Upload GeoJSON / GeoPackage / zipped Shapefile layers with source, licence and notes
- [x] Data-source register (user-supplied vs synthetic, confidence)
- [x] Elevation profile chart; Executive Overview uses the active project's routes

## Phase 3: Cost engine ✅ (2026-10-30)
- [x] Editable cost assumptions (materials, construction/inch-km, terrain multiplier, water crossings,
      wetland premium, forest clearing, land rights, shared-corridor discount, access roads, engineering &
      permitting, environmental mitigation, restoration, contingency, O&M, integrity, energy, life, discount
      rate, decommissioning), saved with the project; status shown as default vs user-edited
- [x] CAPEX, OPEX, lifecycle cost (undiscounted + present value), cost per km, CAPEX vs budget
- [x] Breakdown by cost group and line-by-line basis of estimate
- [x] One-at-a-time sensitivity (tornado) on CAPEX or lifecycle PV
- [x] Cumulative cost over the design life
- [x] Replaced the Phase 1 placeholder rates; route scoring uses CAPEX and lifecycle PV from the cost engine
- [ ] Later: pump/compressor stations from hydraulics, construction spread over several years, inflation,
      Monte Carlo ranges (P10/P50/P90), editable routing-strategy weights

## Phase 4: Environmental & community ✅ (2026-10-30)
- [x] Environmental screening: 11 themes per route with level, data status and confidence
- [x] Data gaps shown as "Not assessed" (never "Low"); new habitat layer category with no synthetic substitute
- [x] Features crossed / nearby per route; mitigation ideas for professional review; CSV export
- [x] Environment routing strategy avoids uploaded sensitive habitat
- [x] Engagement records: activities, concerns, commitments, follow-up actions, land-use & traditional
      activity considerations (no location fields), project changes from engagement
- [x] Sensitivity levels; Restricted hidden by default and excluded from exports; overdue tracking
- [x] Route-linked concerns shown on Route Planning as information only (never scored)
- [x] Executive Overview: environmental & community snapshot
- [ ] Later: editable screening thresholds in the app; mapped floodplain / soils / landslide inventories;
      real access control (user accounts) before any multi-user deployment; file attachments for records

## Phase 5: Digital twin ✅ (2026-10-30)
- [x] Optional real terrain: GeoTIFF DEM upload (rasterio), cropped and stored per project; used by routing,
      profiles, environmental proxies and the twin; unknown where not covered (never filled in)
- [x] Twin built from the project's real coordinates (UTM), with terrain, draped constraint layers and all route options
- [x] Planning / Construction / Operations modes; indicative facilities (station, block valves, terminal)
- [x] Construction stage per section with length-weighted progress, saved with the project
- [x] User chooses the route option to model; Monitoring and the twin follow its length and pipeline type
- [x] Renderer-independent scene (`TwinScene`) with JSON export for other 3D tools (path to Omniverse/USD later)
- [x] Pages show a clear message instead of crashing when project locations are invalid
- [ ] Later: real hydrography/roads from open data services, construction dates & schedule, USD/glTF export

## Phase 6: Monitoring & ML ✅ (2026-10-30)
- [x] New silent-fault scenarios: flow meter bias, pressure sensor drift
- [x] Leak indicator cross-checks downstream pressure (fluid loss vs possible meter fault)
- [x] Replay without look-ahead (events recomputed from data up to the replay time)
- [x] Optional ML detection (scikit-learn): robust covariance (default) or Isolation Forest, baseline-trained,
      per-event explanation of the largest departures, advisory only
- [x] Detector comparison against the simulation's ground truth (detected? delay, false alarms)
- [x] Operator acknowledgement workflow (status, note, name), persisted; never silences rules
- [ ] Later: retrain on rolling windows, operating-mode changes (flow set-point steps), multi-fault scenarios,
      SCADA/historian CSV import for offline analysis

## Phase 7: Reports & integration ✅ (2026-10-30)
- [x] Risk & mitigation register (5 x 5, inherent and residual, owners, mitigation status), stored per project
- [x] Risk suggestions from routes, environmental screening, costs, data gaps and engagement, reviewed by a person
      before entering the register (accepted or set aside; never auto-added or auto-scored)
- [x] Mitigation commitments combined from the risk register and engagement records
- [x] Investor dashboard; PDF report (fpdf2) and CSV bundle with data-class tags on every section/table
- [x] Restricted engagement records excluded from all reports (tested)
- [x] Executive Overview: risk snapshot and links

## Next ideas (beyond the original 7 phases)
- [ ] Put the project under version control (git) and back up `data/`
- [ ] Real open-data connectors (hydrography, roads, protected areas) instead of manual uploads
- [ ] Probabilistic costs (P10/P50/P90) and schedule; stations from hydraulic calculations
- [ ] User accounts and real access control before any multi-user or hosted deployment
- [ ] Charts embedded in the PDF report (would need an image renderer, e.g. matplotlib)

## Known limitations
- The simulation physics is deliberately simple (linear pressure loss, daily cycles, noise).
- ML defaults were tuned on the same simulator they are evaluated on; results are optimistic.
- Terrain is synthetic unless a DEM is uploaded; DEMs are resampled to ~250 m cells for the demo distance.
- Synthetic layers are drawn over a real basemap; they are labelled and dashed but do not match the
  real landscape shown underneath.
- Corridors are screening-level (grid of ~0.8 km cells for the demo distance), not engineering alignments.
- Strategy weights are fixed in `modules/routing.py` (not yet editable in the app).
- Environmental screening thresholds are fixed in `EnvThresholds` (shown, not editable in the app).
- Sensitivity labels are not security: anyone with access to this computer can read the database file.
- Cost defaults are generic illustrative values, not benchmarks; energy, stations, financing and tax are excluded.
- Facility positions (stations, valves) are indicative placements, not hydraulic or code-based design.
