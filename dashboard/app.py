import streamlit as st
import pandas as pd
import plotly.express as px
from sqlalchemy import create_engine
from urllib.parse import quote_plus


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="E-Commerce Sales Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Reduce overall page spacing */
    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 1rem;
        max-width: 1400px;
    }

    /* Main title */
    .main-title {
        font-size: 28px;
        font-weight: 650;
        margin-bottom: 2px;
    }

    .subtitle {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 12px;
    }

    /* Section headings */
    .section-title {
        font-size: 19px;
        font-weight: 600;
        margin-top: 12px;
        margin-bottom: 8px;
    }

    /* Small description */
    .section-text {
        font-size: 13px;
        color: #6b7280;
        margin-bottom: 8px;
    }

    /* KPI cards */
    .kpi-card {
        padding: 13px 15px;
        border-radius: 13px;
        min-height: 105px;
        border: 1px solid rgba(0,0,0,0.06);
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .kpi-icon {
        font-size: 18px;
        margin-bottom: 3px;
    }

    .kpi-title {
        font-size: 12px;
        font-weight: 500;
    }

    .kpi-value {
        font-size: 21px;
        font-weight: 650;
        margin-top: 3px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #faf8ff;
    }

    section[data-testid="stSidebar"] h1 {
        font-size: 21px;
    }

    section[data-testid="stSidebar"] h2 {
        font-size: 17px;
    }

    section[data-testid="stSidebar"] h3 {
        font-size: 15px;
    }

    /* Reduce chart spacing */
    div[data-testid="stPlotlyChart"] {
        margin-top: -5px;
        margin-bottom: 5px;
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        border-radius: 10px;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# DATABASE
# =========================================================

DB_USER = "postgres"
DB_PASSWORD = "Meghana@123"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "ecommerce_dw"

password = quote_plus(DB_PASSWORD)

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{password}@"
    f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


# =========================================================
# LOAD SALES DATA
# =========================================================

@st.cache_data
def load_sales_data():

    query = """
    SELECT
        full_date,
        year,
        month,
        month_name,
        quarter,
        country,
        stock_code,
        description,
        total_orders,
        total_quantity,
        total_revenue,
        average_order_value
    FROM sales_mart
    """

    return pd.read_sql(query, engine)


# =========================================================
# LOAD CUSTOMER DATA
# =========================================================

@st.cache_data
def load_customer_data():

    query = """
    SELECT
        customer_id,
        country,
        total_orders,
        total_quantity,
        total_revenue,
        average_order_value,
        first_purchase_date,
        last_purchase_date,
        customer_lifetime_days,
        purchase_days
    FROM customer_mart
    """

    return pd.read_sql(query, engine)


sales = load_sales_data()
customers = load_customer_data()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🛒 E-Commerce")
st.sidebar.markdown("### Dashboard")

category = st.sidebar.radio(
    "Select Category",
    [
        "Overview",
        "Sales Trends",
        "Products",
        "Customers",
        "Geography"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### Filters")

countries = ["All"] + sorted(
    sales["country"].dropna().unique().tolist()
)

selected_country = st.sidebar.selectbox(
    "🌍 Country",
    countries
)

years = ["All"] + sorted(
    sales["year"].dropna().unique().tolist()
)

selected_year = st.sidebar.selectbox(
    "📅 Year",
    years
)

st.sidebar.divider()

st.sidebar.caption(
    "PostgreSQL • Python • Streamlit • Plotly"
)


# =========================================================
# FILTER SALES
# =========================================================

filtered_sales = sales.copy()

if selected_country != "All":
    filtered_sales = filtered_sales[
        filtered_sales["country"] == selected_country
    ]

if selected_year != "All":
    filtered_sales = filtered_sales[
        filtered_sales["year"] == selected_year
    ]


# =========================================================
# FILTER CUSTOMERS
# =========================================================

filtered_customers = customers.copy()

if selected_country != "All":
    filtered_customers = filtered_customers[
        filtered_customers["country"] == selected_country
    ]

# If a year is selected, show customers who purchased
# during that selected year.
if selected_year != "All":

    customer_ids = filtered_sales["stock_code"].copy()

    period_data = sales[
        sales["year"] == selected_year
    ].copy()

    if selected_country != "All":
        period_data = period_data[
            period_data["country"] == selected_country
        ]

    # Customer IDs are not directly in sales_mart,
    # so for year filtering use the customer mart's
    # country-level population when necessary.
    if selected_country != "All":
        filtered_customers = customers[
            customers["country"] == selected_country
        ]
    else:
        filtered_customers = customers.copy()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🛒 E-Commerce Sales Analytics</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Interactive business intelligence dashboard for retail sales and customer insights'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# KPI VALUES
# =========================================================

total_revenue = filtered_sales["total_revenue"].sum()

total_orders = filtered_sales["total_orders"].sum()

total_quantity = filtered_sales["total_quantity"].sum()

total_products = filtered_sales["stock_code"].nunique()

total_customers = filtered_customers["customer_id"].nunique()


# =========================================================
# KPI CARDS
# =========================================================

c1, c2, c3, c4, c5 = st.columns(5)


with c1:
    st.markdown(
        f"""
        <div class="kpi-card" style="background:#ecfdf5;">
            <div class="kpi-icon">💰</div>
            <div class="kpi-title" style="color:#047857;">
                Revenue
            </div>
            <div class="kpi-value" style="color:#065f46;">
                £{total_revenue:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c2:
    st.markdown(
        f"""
        <div class="kpi-card" style="background:#fff7ed;">
            <div class="kpi-icon">📦</div>
            <div class="kpi-title" style="color:#ea580c;">
                Orders
            </div>
            <div class="kpi-value" style="color:#9a3412;">
                {total_orders:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c3:
    st.markdown(
        f"""
        <div class="kpi-card" style="background:#fdf2f8;">
            <div class="kpi-icon">👥</div>
            <div class="kpi-title" style="color:#db2777;">
                Customers
            </div>
            <div class="kpi-value" style="color:#9d174d;">
                {total_customers:,}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c4:
    st.markdown(
        f"""
        <div class="kpi-card" style="background:#f5f3ff;">
            <div class="kpi-icon">🛍️</div>
            <div class="kpi-title" style="color:#7c3aed;">
                Products
            </div>
            <div class="kpi-value" style="color:#5b21b6;">
                {total_products:,}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


with c5:
    st.markdown(
        f"""
        <div class="kpi-card" style="background:#ecfeff;">
            <div class="kpi-icon">📊</div>
            <div class="kpi-title" style="color:#0891b2;">
                Quantity
            </div>
            <div class="kpi-value" style="color:#155e75;">
                {total_quantity:,}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


st.markdown("<br>", unsafe_allow_html=True)


# =========================================================
# OVERVIEW
# =========================================================

if category == "Overview":

    st.markdown(
        '<div class="section-title">📊 Business Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-text">'
        'Key sales performance and customer insights'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # -----------------------------------------
    # Revenue
    # -----------------------------------------

    with col1:

        monthly = (
            filtered_sales
            .assign(
                month_date=lambda x:
                pd.to_datetime(x["full_date"])
                .dt.to_period("M")
                .dt.to_timestamp()
            )
            .groupby("month_date", as_index=False)
            ["total_revenue"]
            .sum()
            .sort_values("month_date")
        )

        fig = px.area(
            monthly,
            x="month_date",
            y="total_revenue",
            markers=True,
            labels={
                "month_date": "",
                "total_revenue": "Revenue"
            }
        )

        fig.update_traces(
            line=dict(
                color="#10b981",
                width=2.5
            ),
            marker=dict(size=6),
            fillcolor="rgba(16,185,129,0.12)"
        )

        fig.update_layout(
            title="Revenue Trend",
            title_font_size=17,
            height=320,
            template="plotly_white",
            margin=dict(l=10, r=10, t=45, b=10),
            xaxis=dict(
                showgrid=False,
                tickformat="%b %Y"
            ),
            yaxis=dict(
                title=None,
                gridcolor="#eeeeee"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # -----------------------------------------
    # Orders
    # -----------------------------------------

    with col2:

        monthly_orders = (
            filtered_sales
            .assign(
                month_date=lambda x:
                pd.to_datetime(x["full_date"])
                .dt.to_period("M")
                .dt.to_timestamp()
            )
            .groupby("month_date", as_index=False)
            ["total_orders"]
            .sum()
            .sort_values("month_date")
        )

        fig = px.bar(
            monthly_orders,
            x="month_date",
            y="total_orders",
            labels={
                "month_date": "",
                "total_orders": "Orders"
            }
        )

        fig.update_traces(
            marker_color="#f97316"
        )

        fig.update_layout(
            title="Orders Trend",
            title_font_size=17,
            height=320,
            template="plotly_white",
            margin=dict(l=10, r=10, t=45, b=10),
            xaxis=dict(
                showgrid=False,
                tickformat="%b %Y"
            ),
            yaxis=dict(
                title=None,
                gridcolor="#eeeeee"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# SALES TRENDS
# =========================================================

elif category == "Sales Trends":

    st.markdown(
        '<div class="section-title">📈 Sales Trends</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # Revenue by month

    with col1:

        monthly = (
            filtered_sales
            .assign(
                month_date=lambda x:
                pd.to_datetime(x["full_date"])
                .dt.to_period("M")
                .dt.to_timestamp()
            )
            .groupby("month_date", as_index=False)
            ["total_revenue"]
            .sum()
            .sort_values("month_date")
        )

        fig = px.line(
            monthly,
            x="month_date",
            y="total_revenue",
            markers=True,
            labels={
                "month_date": "",
                "total_revenue": "Revenue"
            }
        )

        fig.update_traces(
            line=dict(
                color="#10b981",
                width=3
            ),
            marker=dict(size=7)
        )

        fig.update_layout(
            title="Monthly Revenue",
            title_font_size=17,
            height=330,
            template="plotly_white",
            margin=dict(l=10, r=10, t=45, b=10),
            xaxis=dict(
                showgrid=False,
                tickformat="%b %Y"
            ),
            yaxis=dict(
                title=None,
                gridcolor="#eeeeee"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Orders

    with col2:

        monthly_orders = (
            filtered_sales
            .assign(
                month_date=lambda x:
                pd.to_datetime(x["full_date"])
                .dt.to_period("M")
                .dt.to_timestamp()
            )
            .groupby("month_date", as_index=False)
            ["total_orders"]
            .sum()
            .sort_values("month_date")
        )

        fig = px.bar(
            monthly_orders,
            x="month_date",
            y="total_orders",
            labels={
                "month_date": "",
                "total_orders": "Orders"
            }
        )

        fig.update_traces(
            marker_color="#f97316"
        )

        fig.update_layout(
            title="Monthly Orders",
            title_font_size=17,
            height=330,
            template="plotly_white",
            margin=dict(l=10, r=10, t=45, b=10),
            xaxis=dict(
                showgrid=False,
                tickformat="%b %Y"
            ),
            yaxis=dict(
                title=None,
                gridcolor="#eeeeee"
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Quarter sales

    quarter_sales = (
        filtered_sales
        .groupby(
            ["year", "quarter"],
            as_index=False
        )["total_revenue"]
        .sum()
    )

    quarter_sales["period"] = (
        quarter_sales["year"].astype(str)
        + " Q"
        + quarter_sales["quarter"].astype(str)
    )

    fig = px.bar(
        quarter_sales,
        x="period",
        y="total_revenue",
        labels={
            "period": "Quarter",
            "total_revenue": "Revenue"
        }
    )

    fig.update_traces(
        marker_color="#8b5cf6"
    )

    fig.update_layout(
        title="Quarterly Revenue",
        title_font_size=17,
        height=300,
        template="plotly_white",
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis=dict(showgrid=False),
        yaxis=dict(
            title=None,
            gridcolor="#eeeeee"
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =========================================================
# PRODUCTS
# =========================================================

elif category == "Products":

    st.markdown(
        '<div class="section-title">🏆 Product Analytics</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    # Top products

    with col1:

        top_products = (
            filtered_sales
            .groupby(
                ["stock_code", "description"],
                as_index=False
            )["total_revenue"]
            .sum()
            .sort_values(
                "total_revenue",
                ascending=False
            )
            .head(10)
            .sort_values("total_revenue")
        )

        fig = px.bar(
            top_products,
            x="total_revenue",
            y="description",
            orientation="h",
            labels={
                "description": "Product",
                "total_revenue": "Revenue"
            }
        )

        fig.update_traces(
            marker_color="#8b5cf6"
        )

        fig.update_layout(
            title="Top 10 Products by Revenue",
            title_font_size=17,
            height=400,
            template="plotly_white",
            margin=dict(l=10, r=10, t=45, b=10),
            xaxis=dict(
                title=None,
                gridcolor="#eeeeee"
            ),
            yaxis=dict(
                title=None,
                showgrid=False,
                automargin=True
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Top products by quantity

    with col2:

        top_quantity = (
            filtered_sales
            .groupby(
                ["stock_code", "description"],
                as_index=False
            )["total_quantity"]
            .sum()
            .sort_values(
                "total_quantity",
                ascending=False
            )
            .head(10)
            .sort_values("total_quantity")
        )

        fig = px.bar(
            top_quantity,
            x="total_quantity",
            y="description",
            orientation="h",
            labels={
                "description": "Product",
                "total_quantity": "Quantity"
            }
        )

        fig.update_traces(
            marker_color="#ec4899"
        )

        fig.update_layout(
            title="Top 10 Products by Quantity",
            title_font_size=17,
            height=400,
            template="plotly_white",
            margin=dict(l=10, r=10, t=45, b=10),
            xaxis=dict(
                title=None,
                gridcolor="#eeeeee"
            ),
            yaxis=dict(
                title=None,
                showgrid=False,
                automargin=True
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# =========================================================
# CUSTOMERS
# =========================================================

elif category == "Customers":

    st.markdown(
        '<div class="section-title">👥 Customer Analytics</div>',
        unsafe_allow_html=True
    )

    customer_data = filtered_customers.copy()

    customer_data["segment"] = customer_data[
        "total_orders"
    ].apply(
        lambda x:
        "One-Time Customer"
        if x == 1
        else "Repeat Customer"
    )

    col1, col2 = st.columns(2)

    # Customer segmentation

    with col1:

        segment_counts = (
            customer_data
            .groupby("segment")
            .size()
            .reset_index(name="customers")
        )

        fig = px.pie(
            segment_counts,
            names="segment",
            values="customers",
            hole=0.55,
            color="segment",
            color_discrete_map={
                "Repeat Customer": "#ec4899",
                "One-Time Customer": "#8b5cf6"
            }
        )

        fig.update_traces(
            textposition="inside",
            textinfo="percent+label"
        )

        fig.update_layout(
            title="Customer Segmentation",
            title_font_size=17,
            height=350,
            template="plotly_white",
            margin=dict(l=10, r=10, t=45, b=10)
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Customer revenue

    with col2:

        top_customers = (
            customer_data
            .sort_values(
                "total_revenue",
                ascending=False
            )
            .head(10)
            .sort_values("total_revenue")
        )

        top_customers["customer_label"] = (
            "Customer "
            + top_customers["customer_id"].astype(str)
        )

        fig = px.bar(
            top_customers,
            x="total_revenue",
            y="customer_label",
            orientation="h",
            labels={
                "customer_label": "Customer",
                "total_revenue": "Revenue"
            }
        )

        fig.update_traces(
            marker_color="#10b981"
        )

        fig.update_layout(
            title="Top Customers by Revenue",
            title_font_size=17,
            height=350,
            template="plotly_white",
            margin=dict(l=10, r=10, t=45, b=10),
            xaxis=dict(
                title=None,
                gridcolor="#eeeeee"
            ),
            yaxis=dict(
                title=None,
                showgrid=False
            )
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # Customer table

    st.markdown(
        '<div class="section-title">💰 Customer Details</div>',
        unsafe_allow_html=True
    )

    table = (
        customer_data[
            [
                "customer_id",
                "country",
                "total_orders",
                "total_revenue",
                "average_order_value"
            ]
        ]
        .sort_values(
            "total_revenue",
            ascending=False
        )
        .head(10)
        .copy()
    )

    table["total_revenue"] = table[
        "total_revenue"
    ].map(lambda x: f"£{x:,.2f}")

    table["average_order_value"] = table[
        "average_order_value"
    ].map(lambda x: f"£{x:,.2f}")

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# GEOGRAPHY
# =========================================================

elif category == "Geography":

    st.markdown(
        '<div class="section-title">🌍 Geographic Sales Analysis</div>',
        unsafe_allow_html=True
    )

    country_sales = (
        filtered_sales
        .groupby("country", as_index=False)
        ["total_revenue"]
        .sum()
        .sort_values(
            "total_revenue",
            ascending=False
        )
        .head(15)
        .sort_values("total_revenue")
    )

    fig = px.bar(
        country_sales,
        x="total_revenue",
        y="country",
        orientation="h",
        labels={
            "country": "Country",
            "total_revenue": "Revenue"
        }
    )

    fig.update_traces(
        marker_color="#06b6d4"
    )

    fig.update_layout(
        title="Top 15 Countries by Revenue",
        title_font_size=17,
        height=450,
        template="plotly_white",
        margin=dict(l=10, r=10, t=45, b=10),
        xaxis=dict(
            title=None,
            gridcolor="#eeeeee"
        ),
        yaxis=dict(
            title=None,
            showgrid=False
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Country summary

    country_summary = (
        filtered_sales
        .groupby("country", as_index=False)
        .agg(
            Revenue=("total_revenue", "sum"),
            Orders=("total_orders", "sum"),
            Quantity=("total_quantity", "sum")
        )
        .sort_values(
            "Revenue",
            ascending=False
        )
        .head(10)
    )

    country_summary["Revenue"] = country_summary[
        "Revenue"
    ].map(lambda x: f"£{x:,.2f}")

    st.dataframe(
        country_summary,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#888;
        font-size:11px;
        padding:4px;
    ">
        E-Commerce Sales Analytics • Data Engineering & MLOps
    </div>
    """,
    unsafe_allow_html=True
)