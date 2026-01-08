"""
PE Rollup Intelligence - Home
Landing page for the PE Rollup Intelligence Platform
"""
import streamlit as st

st.set_page_config(
    page_title="PE Rollup Intelligence Platform for EBITDA Lift",
    page_icon="📊",
    layout="wide"
)

# Sidebar toggle button styling - make it prominent
st.markdown("""
<style>
/* Sidebar toggle button - bold and obvious */
button[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] button,
section[data-testid="stSidebar"] + div button:first-child {
    font-size: 1.2rem !important;
    width: 2.4rem !important;
    height: 2.4rem !important;
    min-width: 2.4rem !important;
    min-height: 2.4rem !important;
    background-color: #A78BFA !important;
    border-radius: 10px !important;
    border: 3px solid #8B5CF6 !important;
    cursor: pointer !important;
    box-shadow: 0 4px 12px rgba(167, 139, 250, 0.4) !important;
}

button[data-testid="stSidebarCollapseButton"] svg,
[data-testid="collapsedControl"] svg {
    width: 1.5rem !important;
    height: 1.5rem !important;
    color: #0D1B2A !important;
    stroke: #0D1B2A !important;
    stroke-width: 3px !important;
}

button[data-testid="stSidebarCollapseButton"]:hover,
[data-testid="collapsedControl"] button:hover {
    background-color: #8B5CF6 !important;
    transform: scale(1.05) !important;
    box-shadow: 0 6px 16px rgba(167, 139, 250, 0.6) !important;
}
</style>
""", unsafe_allow_html=True)

# Main content
st.title("PE Rollup Intelligence Platform for EBITDA Lift")
st.markdown("### R-GCN for Private Equity Vendor Consolidation")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ## The Problem

    Private Equity rollups face a **vendor consolidation challenge**:
    when acquiring multiple dental practices, which vendors should be
    standardized across the portfolio to maximize EBITDA?

    ## The Solution

    This platform uses **Relational Graph Convolutional Networks (R-GCN)**
    to model vendor relationships and predict optimal consolidation paths.
    """)

with col2:
    st.markdown("## Model Performance")

    met1, met2 = st.columns(2)
    with met1:
        st.metric("PR-AUC", "0.9407", "+0.37% vs baseline")
        st.metric("Risk MAE", "1.14 days", "< 3.0 target")
    with met2:
        st.metric("Calibration (ECE)", "0.0000", "Perfect")
        st.metric("Clustering", "0.3447", "> 0.30 target")

st.markdown("---")

st.markdown("""
## Navigate

Use the sidebar to explore:

- **Dashboard** - Value creation analysis, What-If scenarios, implementation timeline
- **Graph Topology** - Interactive visualization of R-GCN predictions on the vendor graph

---

**Author:** Gregory E. Schwartz
M.S. Artificial Intelligence (Yeshiva University) | MBA (Cornell University)

[GitHub](https://github.com/ges257/pe-rollup-intelligence-system) |
[Causal Synth Engine](https://github.com/ges257/causal-synth-engine)
""")
