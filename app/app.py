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
