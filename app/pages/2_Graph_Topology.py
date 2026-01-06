"""
PE Rollup Intelligence - Graph Topology Visualizer
Interactive visualization of R-GCN model predictions on Site-Vendor relationships
"""
import streamlit as st
import pandas as pd
from streamlit_agraph import agraph, Node, Edge, Config
from pathlib import Path

st.set_page_config(
    page_title="Graph Topology | PE Rollup Intelligence Platform",
    page_icon="🔗",
    layout="wide"
)

# ============================================================================
# SIDEBAR - Explainer with Model Metrics and Legend
# ============================================================================
with st.sidebar:
    st.markdown("## PE Rollup Intelligence Platform")
    st.markdown("> **R-GCN predictions** visualized on the vendor graph")

    st.markdown("---")
    st.markdown("### Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("PR-AUC", "0.9407", "+0.37%")
        st.metric("Risk MAE", "1.14 days")
    with col2:
        st.metric("ECE", "0.0000", "Perfect")
        st.metric("Silhouette", "0.3447")

    st.markdown("---")
    st.markdown("### What You're Seeing")
    st.markdown("""
    **This graph shows MODEL PREDICTIONS:**
    - Node colors = Predicted risk level
    - Edge thickness = Adoption probability
    - NOT just raw input data
    """)

    st.markdown("---")
    st.markdown("### Legend")
    st.markdown("""
    **Risk Labels (Model Output):**
    - 🟢 **Green** = Low Risk (p > 0.7)
    - 🟡 **Amber** = Medium Risk (0.4 < p < 0.7)
    - 🔴 **Red** = High Risk (p < 0.4)

    **Sites:** Blue circles (portfolio companies)
    """)

    st.markdown("---")
    st.markdown("### Links")
    st.markdown("[GitHub](https://github.com/ges257/pe-rollup-intelligence-system)")
    st.markdown("[Causal Synth Engine](https://github.com/ges257/causal-synth-engine)")

# ============================================================================
# DATA LOADING - Model Predictions
# ============================================================================
@st.cache_data
def load_predictions():
    """Load model predictions from plan_table.csv"""
    try:
        # Try relative path first (for local dev)
        data_path = Path(__file__).parent.parent / "data" / "plan_table.csv"
        if data_path.exists():
            return pd.read_csv(data_path)
        # Fallback for HuggingFace
        return pd.read_csv("data/plan_table.csv")
    except Exception as e:
        st.error(f"Could not load predictions: {e}")
        return None

@st.cache_data
def load_raw_data():
    """Load raw site/vendor data for node attributes"""
    try:
        # Try multiple paths
        for base in [Path(__file__).parent.parent.parent, Path(".")]:
            sites_path = base / "data" / "sites.csv"
            vendors_path = base / "data" / "vendors.csv"
            if sites_path.exists():
                sites = pd.read_csv(sites_path)
                vendors = pd.read_csv(vendors_path)
                return sites, vendors
        return None, None
    except:
        return None, None

predictions_df = load_predictions()
sites_df, vendors_df = load_raw_data()

# ============================================================================
# MAIN CONTENT
# ============================================================================
st.title("R-GCN Prediction Graph")
st.markdown("**Model predictions visualized** — edge thickness = adoption probability, node color = risk")

if predictions_df is not None:
    # Filter Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        regions = ['All'] + sorted(predictions_df['region'].unique().tolist())
        selected_region = st.selectbox("Filter by Region", regions)

    with col2:
        categories = ['All'] + sorted(predictions_df['category'].unique().tolist())
        selected_category = st.selectbox("Filter by Vendor Category", categories)

    with col3:
        quarters = ['All'] + sorted(predictions_df['quarter'].unique().tolist())
        selected_quarter = st.selectbox("Filter by Quarter", quarters)

    # Filter data
    filtered_df = predictions_df.copy()
    if selected_region != 'All':
        filtered_df = filtered_df[filtered_df['region'] == selected_region]
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['category'] == selected_category]
    if selected_quarter != 'All':
        filtered_df = filtered_df[filtered_df['quarter'] == selected_quarter]

    # Limit for readability
    max_edges = st.slider("Max Recommendations to Show", 10, 50, 25)
    filtered_df = filtered_df.head(max_edges)

    # ========================================================================
    # BUILD GRAPH FROM MODEL PREDICTIONS
    # ========================================================================
    nodes = []
    edges = []
    added_sites = set()
    added_vendors = set()

    # Risk colors (from model predictions)
    risk_colors = {
        'Green': "#2ecc71",
        'Amber': "#f1c40f",
        'Red': "#e74c3c"
    }

    # Category colors for vendor nodes
    category_colors = {
        'Lab': '#9b59b6',
        'RCM': '#3498db',
        'Telephony': '#1abc9c',
        'Scheduling': '#e67e22',
        'Clearinghouse': '#34495e',
        'IT_MSP': '#95a5a6',
        'Supplies': '#d35400'
    }

    for _, row in filtered_df.iterrows():
        site_id = row['site_id']
        vendor_id = row['vendor_id']
        site_name = row['site_name']
        vendor_name = row['vendor_name']
        risk_label = row['risk_label']
        p_adoption = row['p_adoption']
        fit_score = row['fit_score']
        category = row['category']
        days_ar = row['days_ar_delta']

        # Add site node (blue)
        if site_id not in added_sites:
            # Get revenue if available
            revenue_str = ""
            if sites_df is not None:
                site_row = sites_df[sites_df['site_id'] == site_id]
                if len(site_row) > 0:
                    revenue_m = site_row.iloc[0]['annual_revenue'] / 1_000_000
                    revenue_str = f"\nRevenue: ${revenue_m:.1f}M"

            nodes.append(Node(
                id=site_id,
                label=f"{site_id}\n({row['region']})",
                size=30,
                color="#3498db",
                title=f"Site: {site_name}\nRegion: {row['region']}\nEHR: {row['ehr_system']}{revenue_str}"
            ))
            added_sites.add(site_id)

        # Add vendor node (colored by PREDICTED RISK)
        if vendor_id not in added_vendors:
            vendor_color = risk_colors.get(risk_label, '#7f8c8d')

            # Get vendor details if available
            tier_str = ""
            if vendors_df is not None:
                vendor_row = vendors_df[vendors_df['vendor_id'] == vendor_id]
                if len(vendor_row) > 0:
                    tier_str = f"\nTier: {vendor_row.iloc[0]['tier']}"

            nodes.append(Node(
                id=vendor_id,
                label=f"{vendor_name[:15]}\n({category})",
                size=22,
                color=vendor_color,
                title=f"Vendor: {vendor_name}\nCategory: {category}{tier_str}\n\n--- MODEL PREDICTIONS ---\nRisk: {risk_label}\nAdoption Prob: {p_adoption:.1%}\nFit Score: {fit_score}"
            ))
            added_vendors.add(vendor_id)

        # Add edge - thickness based on PREDICTED ADOPTION PROBABILITY
        edge_color = risk_colors.get(risk_label, '#bdc3c7')
        edge_width = max(1, p_adoption * 5)  # Scale probability to width

        edges.append(Edge(
            source=site_id,
            target=vendor_id,
            color=edge_color,
            width=edge_width,
            title=f"p(adopt): {p_adoption:.1%}\nFit: {fit_score}\nA/R Impact: {days_ar:+.1f} days"
        ))

    # Graph configuration
    config = Config(
        width=950,
        height=550,
        directed=False,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
        node={'labelProperty': 'label'},
        link={'labelProperty': 'label', 'renderLabel': False}
    )

    # Stats row
    st.markdown("---")
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    with stat_col1:
        st.metric("Sites", len(added_sites))
    with stat_col2:
        st.metric("Vendors", len(added_vendors))
    with stat_col3:
        avg_prob = filtered_df['p_adoption'].mean()
        st.metric("Avg Adoption Prob", f"{avg_prob:.1%}")
    with stat_col4:
        green_pct = (filtered_df['risk_label'] == 'Green').mean() * 100
        st.metric("Green (Low Risk)", f"{green_pct:.0f}%")

    st.markdown("---")

    # Render graph
    if len(nodes) > 0:
        agraph(nodes=nodes, edges=edges, config=config)

        st.markdown("""
        **Hover over nodes/edges** for model prediction details.

        | Visual | Meaning |
        |--------|---------|
        | Edge thickness | Higher = higher predicted adoption probability |
        | Vendor color | 🟢 Green = low risk, 🟡 Amber = medium, 🔴 Red = high risk |
        | Site color | Blue = portfolio company (PortCo) |
        """)
    else:
        st.warning("No predictions match current filters. Try adjusting.")

    # Risk Distribution from MODEL
    st.markdown("---")
    st.markdown("### Model Risk Distribution")

    risk_counts = filtered_df['risk_label'].value_counts()
    col1, col2, col3 = st.columns(3)

    for i, (risk, count) in enumerate(risk_counts.items()):
        cols = [col1, col2, col3]
        with cols[i % 3]:
            pct = count / len(filtered_df) * 100
            emoji = {'Green': '🟢', 'Amber': '🟡', 'Red': '🔴'}.get(risk, '⚪')
            st.metric(f"{emoji} {risk}", f"{count} ({pct:.0f}%)")

    # Fit Score Distribution
    st.markdown("### Fit Score Distribution (Model Output)")
    fit_counts = filtered_df['fit_label'].value_counts()

    fit_order = ['Excellent Fit', 'Good Fit', 'Moderate Fit', 'Poor Fit', 'Not Recommended']
    cols = st.columns(len(fit_order))
    for i, label in enumerate(fit_order):
        with cols[i]:
            count = fit_counts.get(label, 0)
            st.metric(label, count)

else:
    st.error("Could not load model predictions. Ensure plan_table.csv exists in data/")
