"""
Olist Brazilian E-Commerce Analytics Dashboard
Author: Aarohi Mistry
Description: Interactive dashboard analyzing sales, customer geography,
             and product/seller performance for the Olist marketplace.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Olist E-Commerce Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CUSTOM STYLING
# -----------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="metric-container"] label {
        color: #555 !important;
        font-weight: 600;
    }
    .insight-box {
        background-color: #eef5ff;
        border-left: 4px solid #1f4e79;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------
# DATA LOADING (CACHED)
# -----------------------------
# The data folder is one level up from this dashboard folder
DATA_DIR = Path(__file__).parent.parent / "data"


@st.cache_data
def load_data():
    """Load and preprocess all Olist datasets."""
    orders = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")
    order_items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")
    customers = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")
    products = pd.read_csv(DATA_DIR / "olist_products_dataset.csv")
    sellers = pd.read_csv(DATA_DIR / "olist_sellers_dataset.csv")
    reviews = pd.read_csv(DATA_DIR / "olist_order_reviews_dataset.csv")

    # Convert timestamps
    orders["order_purchase_timestamp"] = pd.to_datetime(orders["order_purchase_timestamp"])
    orders["year_month"] = orders["order_purchase_timestamp"].dt.strftime("%Y-%m")

    # Filter to delivered orders within 2017-2018 (matches notebook scope)
    delivered = orders[orders["order_status"] == "delivered"].copy()
    delivered_1718 = delivered[delivered["year_month"].str.startswith(("2017", "2018"))].copy()

    return {
        "orders": orders,
        "order_items": order_items,
        "customers": customers,
        "products": products,
        "sellers": sellers,
        "reviews": reviews,
        "delivered": delivered,
        "delivered_1718": delivered_1718,
    }


@st.cache_data
def compute_revenue_data(_data):
    """Compute monthly revenue (delivered orders, 2017-2018)."""
    merged = pd.merge(
        _data["delivered"][["order_id", "year_month"]],
        _data["order_items"][["order_id", "price"]],
        on="order_id", how="inner"
    )
    monthly = merged.groupby("year_month", as_index=False)["price"].sum()
    monthly = monthly[monthly["year_month"].str.startswith(("2017", "2018"))]
    monthly = monthly.sort_values("year_month").reset_index(drop=True)
    monthly = monthly.rename(columns={"price": "revenue"})
    return monthly


@st.cache_data
def compute_state_data(_data):
    """Compute customers and revenue per state."""
    orders_customers = pd.merge(
        _data["delivered_1718"][["order_id", "customer_id"]],
        _data["customers"][["customer_id", "customer_state", "customer_unique_id"]],
        on="customer_id", how="inner"
    )
    rev_state = pd.merge(
        orders_customers,
        _data["order_items"][["order_id", "price"]],
        on="order_id", how="inner"
    )
    cust_per_state = (
        orders_customers.groupby("customer_state")["customer_unique_id"]
        .nunique().sort_values(ascending=False).reset_index()
        .rename(columns={"customer_unique_id": "customers"})
    )
    rev_per_state = (
        rev_state.groupby("customer_state")["price"]
        .sum().sort_values(ascending=False).reset_index()
        .rename(columns={"price": "revenue"})
    )
    return cust_per_state, rev_per_state


@st.cache_data
def compute_category_data(_data):
    """Compute revenue and review score by product category."""
    orders_items = pd.merge(
        _data["delivered_1718"][["order_id"]],
        _data["order_items"][["order_id", "product_id", "price", "seller_id"]],
        on="order_id", how="inner"
    )
    items_products = pd.merge(
        orders_items,
        _data["products"][["product_id", "product_category_name"]],
        on="product_id", how="inner"
    )
    cat_revenue = (
        items_products.groupby("product_category_name")["price"]
        .sum().sort_values(ascending=False).reset_index()
        .rename(columns={"price": "revenue"})
    )

    items_reviews = pd.merge(
        orders_items,
        _data["reviews"][["order_id", "review_score"]],
        on="order_id", how="inner"
    )
    reviews_with_cat = pd.merge(
        items_reviews,
        _data["products"][["product_id", "product_category_name"]],
        on="product_id", how="inner"
    )
    cat_review = (
        reviews_with_cat.groupby("product_category_name")["review_score"]
        .agg(["mean", "count"]).reset_index()
        .rename(columns={"mean": "avg_review_score", "count": "review_count"})
    )
    cat_review = cat_review[cat_review["review_count"] >= 50].sort_values(
        "avg_review_score", ascending=False
    )
    return cat_revenue, cat_review


@st.cache_data
def compute_seller_data(_data):
    """Compute revenue and order count per seller."""
    orders_items = pd.merge(
        _data["delivered_1718"][["order_id"]],
        _data["order_items"][["order_id", "seller_id", "price"]],
        on="order_id", how="inner"
    )
    seller_summary = (
        orders_items.groupby("seller_id")
        .agg(total_revenue=("price", "sum"), orders=("order_id", "nunique"))
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )
    return seller_summary


@st.cache_data
def compute_price_review(_data):
    """Compute price vs review score for delivered orders."""
    pr = pd.merge(
        _data["order_items"][["order_id", "price"]],
        _data["reviews"][["order_id", "review_score"]],
        on="order_id", how="inner"
    )
    delivered_ids = _data["delivered_1718"]["order_id"]
    pr = pr[pr["order_id"].isin(delivered_ids)]
    pr = pr[["price", "review_score"]].dropna()
    return pr


# -----------------------------
# CHECK DATA AVAILABILITY
# -----------------------------
required_files = [
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_order_reviews_dataset.csv",
]
missing = [f for f in required_files if not (DATA_DIR / f).exists()]

if missing:
    st.error("⚠️ **Dataset files not found.** Please add the following CSV files to the `data/` folder:")
    for f in missing:
        st.markdown(f"- `{f}`")
    st.markdown(f"""
    **Looking for data in:** `{DATA_DIR.resolve()}`

    **How to get the data:**
    1. Go to [Kaggle: Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
    2. Download the dataset
    3. Place all CSV files inside the `data/` folder
    4. Refresh this page
    """)
    st.stop()

# Load the data
with st.spinner("Loading data..."):
    data = load_data()
    monthly_revenue = compute_revenue_data(data)
    cust_per_state, rev_per_state = compute_state_data(data)
    cat_revenue, cat_review = compute_category_data(data)
    seller_summary = compute_seller_data(data)
    price_review = compute_price_review(data)


# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("Olist Ecommerce Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    [
        "🏠 Overview",
        "💰 Sales Performance",
        "🌎 Customer Geography",
        "📦 Product & Seller Insights",
        "🔑 Key Insights",
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("**About this project**")
st.sidebar.info(
    "End-to-end business analysis of the Brazilian Olist marketplace, "
    "covering sales, geography, and product/seller dynamics for Jan 2017 – Aug 2018."
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Author:** Aarohi Mistry")
st.sidebar.markdown(
    "[GitHub](https://github.com/AarohiAnalyzes/olist-ecommerce-analysis) · "
    "[LinkedIn](https://linkedin.com/in/aarohi-mistry-715713219)"
)


# -----------------------------
# PAGE: OVERVIEW
# -----------------------------
if page == "🏠 Overview":
    st.markdown('<p class="main-header">Olist Brazilian E-Commerce Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">End-to-end business analysis of marketplace sales, customer geography, and product/seller dynamics (Jan 2017 – Aug 2018)</p>', unsafe_allow_html=True)

    total_revenue = monthly_revenue["revenue"].sum()
    total_orders = data["delivered_1718"]["order_id"].nunique()
    total_customers = data["delivered_1718"]["customer_id"].nunique()
    total_sellers = data["sellers"]["seller_id"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"R$ {total_revenue/1e6:,.2f}M")
    c2.metric("Delivered Orders", f"{total_orders:,}")
    c3.metric("Unique Customers", f"{total_customers:,}")
    c4.metric("Active Sellers", f"{total_sellers:,}")

    st.markdown("---")
    st.markdown("### Project Overview")
    st.markdown("""
    This dashboard explores the Olist Brazilian e-commerce marketplace dataset to answer three core business questions:

    1. **Sales Performance**: How does revenue evolve over time? Are there seasonal patterns?
    2. **Customer Geography**: Where are customers concentrated? Which states drive revenue?
    3. **Product & Seller Insights**: Which categories dominate? How do seller strategies differ? Does price drive satisfaction?

    Use the sidebar to navigate between sections.
    """)

    st.markdown("### Methodology")
    st.markdown("""
    - **Scope:** Delivered orders only, January 2017 - August 2018 (20 months)
    - **Tools:** Python (Pandas, NumPy), Plotly for interactive visualization
    - **Pipeline:** 6 relational tables joined via SQL-style merges
    - **Total records analyzed:** ~96K orders, ~99K customers, ~3K sellers
    """)


# -----------------------------
# PAGE: SALES PERFORMANCE
# -----------------------------
elif page == "💰 Sales Performance":
    st.markdown('<p class="main-header">Sales Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Monthly revenue trends and seasonality (Jan 2017 - Aug 2018)</p>', unsafe_allow_html=True)

    total_revenue = monthly_revenue["revenue"].sum()
    avg_monthly = monthly_revenue["revenue"].mean()
    best = monthly_revenue.loc[monthly_revenue["revenue"].idxmax()]
    worst = monthly_revenue.loc[monthly_revenue["revenue"].idxmin()]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"R$ {total_revenue/1e6:,.2f}M")
    c2.metric("Avg. Monthly", f"R$ {avg_monthly/1e3:,.0f}K")
    c3.metric("Best Month", f"{best['year_month']}", f"R$ {best['revenue']/1e3:,.0f}K")
    c4.metric("Worst Month", f"{worst['year_month']}", f"R$ {worst['revenue']/1e3:,.0f}K")

    st.markdown("---")

    fig = px.line(
        monthly_revenue, x="year_month", y="revenue",
        markers=True,
        title="Monthly Revenue (Delivered Orders, Jan 2017 - Aug 2018)",
        labels={"year_month": "Month", "revenue": "Revenue (R$)"},
    )
    fig.update_traces(line=dict(color="#1f4e79", width=3), marker=dict(size=8))
    fig.update_layout(
        height=450,
        xaxis_tickangle=-45,
        plot_bgcolor="white",
        title_font_size=18,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <strong>📌 Key Insight:</strong> Revenue peaked in <strong>November 2017</strong>, driven by the Black Friday effect.
    The marketplace shows strong upward momentum from Jan 2017 onwards, with growing volume month-over-month.
    A clear seasonality pattern emerges - Q4(Oct|Nov|Dec) consistently outperforms other quarters.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📋 View monthly revenue data"):
        display_df = monthly_revenue.copy()
        display_df["revenue"] = display_df["revenue"].apply(lambda x: f"R$ {x:,.2f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# -----------------------------
# PAGE: CUSTOMER GEOGRAPHY
# -----------------------------
elif page == "🌎 Customer Geography":
    st.markdown('<p class="main-header">Customer Geographic Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Where are Olist customers concentrated, and which states drive revenue?</p>', unsafe_allow_html=True)

    total_states = cust_per_state["customer_state"].nunique()
    top_state_cust = cust_per_state.iloc[0]
    top_state_rev = rev_per_state.iloc[0]
    cust_concentration = (cust_per_state.head(3)["customers"].sum() / cust_per_state["customers"].sum()) * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("States Covered", f"{total_states}")
    c2.metric("Top State (Customers)", top_state_cust["customer_state"], f"{top_state_cust['customers']:,}")
    c3.metric("Top State (Revenue)", top_state_rev["customer_state"], f"R$ {top_state_rev['revenue']/1e6:,.2f}M")
    c4.metric("Top 3 States Share", f"{cust_concentration:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        top_cust = cust_per_state.head(10)
        fig = px.bar(
            top_cust, x="customer_state", y="customers",
            title="Top 10 States by Number of Customers",
            labels={"customer_state": "State", "customers": "Unique Customers"},
            color="customers", color_continuous_scale="Blues",
        )
        fig.update_layout(height=400, plot_bgcolor="white", showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_rev = rev_per_state.head(10)
        fig = px.bar(
            top_rev, x="customer_state", y="revenue",
            title="Top 10 States by Revenue",
            labels={"customer_state": "State", "revenue": "Revenue (R$)"},
            color="revenue", color_continuous_scale="Greens",
        )
        fig.update_layout(height=400, plot_bgcolor="white", showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    combined = pd.merge(cust_per_state, rev_per_state, on="customer_state").head(10)
    combined["revenue_M"] = combined["revenue"] / 1e6
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=combined["customer_state"], y=combined["customers"],
        name="Customers", marker_color="#1f4e79", yaxis="y"
    ))
    fig.add_trace(go.Scatter(
        x=combined["customer_state"], y=combined["revenue_M"],
        name="Revenue (R$ M)", line=dict(color="#d62728", width=3),
        marker=dict(size=10), yaxis="y2"
    ))
    fig.update_layout(
        title="Customers vs. Revenue Across Top 10 States",
        height=400,
        plot_bgcolor="white",
        yaxis=dict(title="Customers"),
        yaxis2=dict(title="Revenue (R$ M)", overlaying="y", side="right"),
        legend=dict(x=0.7, y=1.1, orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <strong>📌 Key Insight:</strong> The Brazilian e-commerce market is highly concentrated in the
    <strong>Southeast region.</strong> São Paulo (SP) alone represents the majority of both customers and revenue.
    This concentration creates both opportunity (deepening engagement in core markets) and risk (over-reliance on a few states).
    </div>
    """, unsafe_allow_html=True)


# -----------------------------
# PAGE: PRODUCT & SELLER
# -----------------------------
elif page == "📦 Product & Seller Insights":
    st.markdown('<p class="main-header">Product & Seller Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Which categories drive revenue, how do sellers compete, and does price drive satisfaction?</p>', unsafe_allow_html=True)

    total_cats = cat_revenue["product_category_name"].nunique()
    top_cat = cat_revenue.iloc[0]
    total_sellers = seller_summary.shape[0]
    top_seller = seller_summary.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Product Categories", f"{total_cats}")
    c2.metric("Top Category", top_cat["product_category_name"][:20], f"R$ {top_cat['revenue']/1e6:,.2f}M")
    c3.metric("Active Sellers", f"{total_sellers:,}")
    c4.metric("Top Seller Revenue", f"R$ {top_seller['total_revenue']/1e3:,.0f}K", f"{top_seller['orders']} orders")

    st.markdown("---")

    st.markdown("### Top 10 Product Categories by Revenue")
    top_cats = cat_revenue.head(10)
    fig = px.bar(
        top_cats, x="revenue", y="product_category_name",
        orientation="h",
        labels={"revenue": "Revenue (R$)", "product_category_name": "Category"},
        color="revenue", color_continuous_scale="Tealgrn",
    )
    fig.update_layout(height=450, plot_bgcolor="white", coloraxis_showscale=False,
                      yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Average Review Score by Category")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 10 (Highest Rated)**")
        top_rev_cats = cat_review.head(10)
        fig = px.bar(
            top_rev_cats, x="avg_review_score", y="product_category_name",
            orientation="h", color="avg_review_score",
            color_continuous_scale="Greens",
            labels={"avg_review_score": "Avg Review Score", "product_category_name": ""},
        )
        fig.update_layout(height=400, plot_bgcolor="white", coloraxis_showscale=False,
                          yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Bottom 10 (Lowest Rated)**")
        bot_rev_cats = cat_review.tail(10)
        fig = px.bar(
            bot_rev_cats, x="avg_review_score", y="product_category_name",
            orientation="h", color="avg_review_score",
            color_continuous_scale="Reds_r",
            labels={"avg_review_score": "Avg Review Score", "product_category_name": ""},
        )
        fig.update_layout(height=400, plot_bgcolor="white", coloraxis_showscale=False,
                          yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Top 10 Sellers by Revenue")
    top_sell = seller_summary.head(10).copy()
    top_sell["seller_short"] = top_sell["seller_id"].str[:8] + "…"
    fig = px.bar(
        top_sell, x="seller_short", y="total_revenue",
        labels={"seller_short": "Seller (ID)", "total_revenue": "Revenue (R$)"},
        color="orders", color_continuous_scale="Viridis",
        hover_data={"seller_short": False, "seller_id": True, "orders": True, "total_revenue": ":,.2f"},
    )
    fig.update_layout(height=400, plot_bgcolor="white",
                      coloraxis_colorbar=dict(title="Orders"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
    <strong>📌 Key Insight:</strong> Top sellers follow <strong>two distinct strategies</strong>:
    <em>premium</em> (fewer, higher-value orders) and <em>volume</em> (many lower-value orders).
    Both can lead to top revenue rankings. Olist's marketplace supports diverse seller business models.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Price vs. Customer Satisfaction")
    corr = price_review["price"].corr(price_review["review_score"])

    sample = price_review.sample(min(5000, len(price_review)), random_state=42)
    fig = px.scatter(
        sample, x="price", y="review_score",
        opacity=0.4, log_x=True,
        labels={"price": "Price (R$, log scale)", "review_score": "Review Score"},
        color_discrete_sequence=["steelblue"],
    )
    fig.update_layout(height=400, plot_bgcolor="white",
                      title=f"Pearson Correlation: {corr:.4f}")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
    <div class="insight-box">
    <strong>📌 Key Insight:</strong> The correlation between price and review score is essentially
    <strong>zero ({corr:.3f})</strong>. <strong>Price does not drive customer satisfaction.</strong>
    Factors like delivery speed, product quality, and seller reliability matter far more.
    </div>
    """, unsafe_allow_html=True)


# -----------------------------
# PAGE: KEY INSIGHTS
# -----------------------------
elif page == "🔑 Key Insights":
    st.markdown('<p class="main-header">Key Business Insights</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Synthesized findings from the Olist marketplace analysis</p>', unsafe_allow_html=True)

    st.markdown("### 💰 Sales Performance")
    st.markdown("""
    - **Total revenue** of ~R$ 13M generated across delivered orders (Jan 2017 - Aug 2018)
    - Strong **upward growth** through 2017, reflecting marketplace expansion
    - **November 2017 peak**: clear Black Friday seasonal effect
    - Marketplace dominated by **single-item orders** (avg ~1.13 items/order)
    """)

    st.markdown("### 🌎 Customer Geography")
    st.markdown("""
    - Customers concentrated in the **Southeast region** of Brazil
    - **São Paulo (SP)** dominates both customer count and total revenue
    - Top 3 states (SP, RJ, MG) account for the majority of marketplace activity
    - Significant **expansion opportunity** in Northern and Northeastern states
    """)

    st.markdown("### 📦 Product Categories")
    st.markdown("""
    - Revenue is concentrated in a few **dominant categories** (top 10 capture most of total revenue)
    - Highest-rated categories tend to be **niche/premium** with smaller order volumes
    - Lowest-rated categories often involve **complex logistics** (e.g., furniture, large items)
    """)

    st.markdown("### 🏪 Seller Dynamics")
    st.markdown("""
    - Top sellers split into two distinct strategies:
        - **Premium model:** fewer orders, higher average order value
        - **Volume model:** many smaller orders, lower margins
    - Both strategies can produce top-tier revenue performance
    """)

    st.markdown("### 💡 Counter-Intuitive Finding")
    st.markdown("""
    - **Price does NOT correlate with customer satisfaction** (Pearson r ≈ 0)
    - Customers don't reward higher prices with higher reviews: what matters is **delivery, quality, and seller reliability**
    - **Strategic implication:** Olist should invest in logistics and seller quality rather than premium pricing strategies
    """)

    st.markdown("---")
    st.markdown("### 🎯 Recommendations")
    st.success("""
    1. **Geographic expansion strategy:** Target underserved Northern/Northeastern states with localized seller programs
    2. **Logistics investment:** Improve delivery in low-rated categories (furniture, oversized items)
    3. **Seller diversity:** Continue supporting both premium and volume sellers: both win
    4. **Q4 readiness:** Build dedicated Black Friday infrastructure given proven seasonal demand
    5. **Customer experience focus:** Prioritize service quality over price-based competition
    """)