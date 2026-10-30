"""FlowGuard: application entry point.

Run with:  streamlit run app.py

This file only sets up the page frame, shared sidebar and navigation.
Each page lives in `app_pages/`, and all calculations live in `modules/`.
"""

from pathlib import Path

import streamlit as st

from modules import ui
from modules.config import APP_NAME

ASSETS = Path(__file__).parent / "assets"

st.set_page_config(page_title=APP_NAME, page_icon=str(ASSETS / "logo_icon.svg"), layout="wide")
# Product logo: full logo at the top of the sidebar, icon only when the sidebar is collapsed.
st.logo(str(ASSETS / "logo.svg"), icon_image=str(ASSETS / "logo_icon.svg"), size="large")

ui.init_state()
ui.inject_css()

pages = {
    "Overview": [
        st.Page("app_pages/0_Executive_Overview.py", title="Executive Overview", icon=":material/dashboard:", default=True),
        st.Page("app_pages/1_Project_Setup.py", title="Project Setup", icon=":material/edit_note:"),
    ],
    "Planning": [
        st.Page("app_pages/2_Route_Planning.py", title="Route Planning", icon=":material/route:"),
        st.Page("app_pages/3_Cost_Optimizer.py", title="Cost Optimizer", icon=":material/payments:"),
        st.Page("app_pages/4_Environmental_Intelligence.py", title="Environmental Intelligence", icon=":material/forest:"),
        st.Page("app_pages/5_Community_Engagement.py", title="Indigenous & Community Engagement", icon=":material/diversity_3:"),
    ],
    "Operations": [
        st.Page("app_pages/6_Digital_Twin.py", title="3D Digital Twin", icon=":material/view_in_ar:"),
        st.Page("app_pages/7_Pipeline_Monitoring.py", title="FlowGuard Monitoring", icon=":material/monitor_heart:"),
    ],
    "Governance": [
        st.Page("app_pages/8_Risk_Register.py", title="Risk & Mitigation Register", icon=":material/fact_check:"),
        st.Page("app_pages/9_Investor_Reports.py", title="Reports & Export", icon=":material/summarize:"),
    ],
}

navigation = st.navigation(pages)
ui.sidebar_controls()
navigation.run()
