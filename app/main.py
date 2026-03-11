# ============================================================
# app/main.py — Sales Analytics & Forecast Streamlit App
# ============================================================

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import joblib
import warnings
warnings.filterwarnings("ignore")

from src.model import FEATURES, forecast_30_days

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0F172A; color: #F1F5F9; font-size: 16px; }
    [data-testid="stSidebar"] { background-color: #1E293B; }
    [data-testid="stSidebar"] p { font-size: 15px !important; }

    [data-testid="stMetric"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 20px 24px;
    }
    [data-testid="stMetricLabel"] { color: #94A3B8 !important; font-size: 15px !important; }
    [data-testid="stMetricValue"] { color: #F1F5F9 !important; font-size: 32px !important; font-weight: 700 !important; }

    h1 { color: #F1F5F9 !important; font-size: 36px !important; }
    h2 { color: #F1F5F9 !important; font-size: 26px !important; }
    h3 { color: #F1F5F9 !important; font-size: 22px !important; }
    p  { font-size: 16px !important; color: #CBD5E1; }

    .rec-card {
        background-color: #1E293B;
        border-left: 5px solid #2563EB;
        border-radius: 10px;
        padding: 28px 32px;
        margin-bottom: 20px;
    }
    .rec-card.warning { border-left-color: #F59E0B; }
    .rec-card.success { border-left-color: #10B981; }
    .rec-card.danger  { border-left-color: #EF4444; }
    .rec-card.info    { border-left-color: #6366F1; }

    .rec-title { font-weight: 700; font-size: 22px; color: #F1F5F9; margin-bottom: 10px; }
    .rec-text  { font-size: 17px; color: #94A3B8; line-height: 1.8; }

    hr { border-color: #334155; margin: 24px 0; }

    .stDownloadButton button {
        background-color: #2563EB !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)


# ── LOAD DATA & MODEL ─────────────────────────────────────────
@st.cache_data
def load_data():
    df_raw      = pd.read_csv("data/processed/retail_store_sales_clean.csv")
    df_features = pd.read_csv("data/processed/retail_store_sales_features.csv")
    df_raw["Transaction Date"] = pd.to_datetime(df_raw["Transaction Date"])
    df_features["date"]        = pd.to_datetime(df_features["date"])
    return df_raw, df_features

@st.cache_resource
def load_model():
    return joblib.load("models/forecast_model_v1.pkl")

df_raw, df_features = load_data()
model = load_model()

# ── GLOBAL VARIABLES ─────────────────────────────────────────
top_category    = df_raw.groupby("Category")["Total Spent"].sum().idxmax()
bottom_category = df_raw.groupby("Category")["Total Spent"].sum().idxmin()
avg_order_value = df_raw["Total Spent"].mean()
total_orders    = len(df_raw)
day_names       = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df_dow          = df_features.copy()
df_dow["dayofweek"] = pd.to_datetime(df_dow["date"]).dt.dayofweek
best_day        = day_names[df_dow.groupby("dayofweek")["revenue"].mean().idxmax()]


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Sales Analytics")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Overview", "Forecast & Recommendations"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(f"**Dataset:** {len(df_raw):,} transactions")
    st.markdown(f"**Period:** {df_raw['Transaction Date'].min().strftime('%b %Y')} — {df_raw['Transaction Date'].max().strftime('%b %Y')}")
    st.markdown(f"**Categories:** {df_raw['Category'].nunique()}")
    st.markdown(f"**Customers:** {df_raw['Customer ID'].nunique()}")


# ════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Sales Overview")
    st.markdown("A high-level summary of store performance across all categories and customers.")
    st.markdown("---")

    daily         = df_features[["date", "revenue"]].copy()
    total_revenue = df_raw["Total Spent"].sum()
    avg_daily     = daily["revenue"].mean()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue",      f"${total_revenue:,.0f}")
    col2.metric("Avg Daily Revenue",  f"${avg_daily:,.0f}")
    col3.metric("Total Transactions", f"{total_orders:,}")
    col4.metric("Top Category",       top_category)

    st.markdown("---")

    st.subheader("Daily Revenue Trend")
    st.markdown("7-day and 30-day rolling averages smooth out daily noise and reveal the underlying trend.")

    rolling_7  = daily["revenue"].rolling(7).mean()
    rolling_30 = daily["revenue"].rolling(30).mean()

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=daily["date"], y=daily["revenue"],
        name="Daily Revenue",
        line=dict(color="#334155", width=1),
        fill="tozeroy", fillcolor="rgba(37,99,235,0.05)"
    ))
    fig1.add_trace(go.Scatter(
        x=daily["date"], y=rolling_7,
        name="7-Day Avg",
        line=dict(color="#2563EB", width=2.5)
    ))
    fig1.add_trace(go.Scatter(
        x=daily["date"], y=rolling_30,
        name="30-Day Avg",
        line=dict(color="#10B981", width=2, dash="dash")
    ))
    fig1.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font=dict(color="#94A3B8", size=13),
        xaxis=dict(gridcolor="#1E293B", showgrid=True),
        yaxis=dict(gridcolor="#1E293B", showgrid=True, tickprefix="$"),
        legend=dict(bgcolor="#1E293B", bordercolor="#334155", borderwidth=1, font=dict(size=13)),
        height=400, margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    st.subheader("Revenue by Category")
    st.markdown("Total revenue generated per product category over the entire period.")

    cat_revenue = df_raw.groupby("Category")["Total Spent"].sum().sort_values(ascending=True).reset_index()

    fig2 = go.Figure(go.Bar(
        x=cat_revenue["Total Spent"],
        y=cat_revenue["Category"],
        orientation="h",
        marker=dict(
            color=cat_revenue["Total Spent"],
            colorscale=[[0, "#1E3A8A"], [1, "#2563EB"]],
            showscale=False
        ),
        text=[f"${v:,.0f}" for v in cat_revenue["Total Spent"]],
        textposition="outside",
        textfont=dict(color="#94A3B8", size=13)
    ))
    fig2.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font=dict(color="#94A3B8", size=13),
        xaxis=dict(gridcolor="#1E293B", tickprefix="$"),
        yaxis=dict(gridcolor="#1E293B", tickfont=dict(size=13)),
        height=420, margin=dict(l=0, r=100, t=20, b=0)
    )
    st.plotly_chart(fig2, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# PAGE 2 — FORECAST & RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════
elif page == "Forecast & Recommendations":
    st.title("30-Day Revenue Forecast")
    st.markdown("Machine learning forecast using **LightGBM** trained on 3 years of daily sales data. Three scenarios reflect realistic uncertainty in the prediction.")
    st.markdown("---")

    forecast_df = forecast_30_days(model, df_features)

    total_realistic   = forecast_df["forecast"].sum()
    total_optimistic  = forecast_df["optimistic"].sum()
    total_pessimistic = forecast_df["pessimistic"].sum()
    avg_daily_forecast = forecast_df["forecast"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("📉 Pessimistic (30d)", f"${total_pessimistic:,.0f}", "-15%")
    col2.metric("🎯 Realistic (30d)",   f"${total_realistic:,.0f}",   "Base forecast")
    col3.metric("📈 Optimistic (30d)",  f"${total_optimistic:,.0f}",  "+15%")

    st.markdown("---")

    st.subheader("Revenue Forecast with Scenarios")
    st.markdown("The shaded area represents the range between pessimistic and optimistic scenarios. The solid blue line is the base forecast.")

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=pd.concat([forecast_df["date"], forecast_df["date"][::-1]]),
        y=pd.concat([forecast_df["optimistic"], forecast_df["pessimistic"][::-1]]),
        fill="toself", fillcolor="rgba(37,99,235,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Scenario Range", showlegend=True
    ))
    fig3.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["pessimistic"],
        name="Pessimistic −15%",
        line=dict(color="#EF4444", width=1.5, dash="dot")
    ))
    fig3.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["optimistic"],
        name="Optimistic +15%",
        line=dict(color="#10B981", width=1.5, dash="dot")
    ))
    fig3.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["forecast"],
        name="Realistic (Base)",
        line=dict(color="#2563EB", width=3)
    ))
    fig3.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font=dict(color="#94A3B8", size=13),
        xaxis=dict(gridcolor="#1E293B"),
        yaxis=dict(gridcolor="#1E293B", tickprefix="$"),
        legend=dict(bgcolor="#1E293B", bordercolor="#334155", borderwidth=1, font=dict(size=13)),
        height=430, margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    st.subheader("Business Recommendations")
    st.markdown("Automated insights based on forecast results, historical trends, and business rules.")
    st.markdown("")

    last_30_avg = df_features["revenue"].tail(30).mean()
    change_pct  = (avg_daily_forecast - last_30_avg) / last_30_avg * 100
    recommendations = []

    # Rule 1 — Overall trend
    if change_pct < -10:
        recommendations.append(("danger", "⚠️ Revenue Decline Expected",
            f"The forecast shows a <strong>{abs(change_pct):.1f}% drop</strong> compared to the last 30 days average of ${last_30_avg:,.0f}/day. "
            "This is a significant signal that requires immediate attention. "
            "Consider launching a time-limited promotional campaign, offering bundle discounts, "
            "or activating re-engagement emails to bring back inactive customers. "
            "Review whether any external factors (seasonality, competition) may be contributing to the decline."))
    elif change_pct > 10:
        recommendations.append(("success", "✅ Strong Revenue Growth Expected",
            f"The forecast shows a <strong>{change_pct:.1f}% increase</strong> compared to the last 30 days average of ${last_30_avg:,.0f}/day. "
            "This is a great opportunity to maximize revenue. "
            "Ensure top-selling products are well-stocked to avoid missed sales. "
            "Consider upselling premium items to customers who are already in a buying mindset. "
            "This is also a good time to test new products or categories with lower risk."))
    else:
        recommendations.append(("info", "📊 Revenue Stable — Focus on Optimization",
            f"The forecast is within <strong>{abs(change_pct):.1f}%</strong> of the last 30 days average (${last_30_avg:,.0f}/day). "
            "Revenue is stable, which means this is the right time to focus on improving margins rather than volume. "
            "Look for upsell opportunities with existing customers, optimize pricing on high-demand items, "
            "and analyze which customer segments have the highest lifetime value to prioritize retention efforts."))

    # Rule 2 — Pessimistic scenario risk
    if total_pessimistic < last_30_avg * 30 * 0.80:
        recommendations.append(("warning", "🔔 Significant Downside Risk — Prepare a Contingency Plan",
            f"Even in the pessimistic scenario, revenue could drop to <strong>${total_pessimistic:,.0f}</strong> over 30 days, "
            f"which is more than 20% below recent performance. "
            "It is advisable to review your variable cost structure and identify areas where spending can be reduced quickly if needed. "
            "Consider building a cash reserve or delaying non-critical investments until the outlook becomes clearer."))

    # Rule 3 — Weekend vs weekday
    forecast_df["is_weekend"] = pd.to_datetime(forecast_df["date"]).dt.dayofweek.isin([5, 6])
    weekend_avg = forecast_df[forecast_df["is_weekend"]]["forecast"].mean()
    weekday_avg = forecast_df[~forecast_df["is_weekend"]]["forecast"].mean()

    if weekend_avg > weekday_avg * 1.1:
        recommendations.append(("success", "📅 Weekends Are Your Peak Days — Capitalize on Them",
            f"Weekend revenue (avg <strong>${weekend_avg:,.0f}/day</strong>) is forecasted to outperform weekdays (avg ${weekday_avg:,.0f}/day). "
            "Consider running weekend flash sales, social media promotions on Fridays, "
            "and ensuring maximum staff availability and stock levels going into the weekend."))
    elif weekday_avg > weekend_avg * 1.1:
        recommendations.append(("info", "📅 Weekdays Drive Your Revenue",
            f"Weekday revenue (avg <strong>${weekday_avg:,.0f}/day</strong>) outperforms weekends (avg ${weekend_avg:,.0f}/day). "
            "Your customers are likely professionals or B2B buyers. "
            "Focus your marketing budget on Monday–Thursday campaigns for maximum impact. "
            "Consider offering weekday-only deals to reinforce this pattern and build loyalty."))
    else:
        recommendations.append(("info", "📅 Consistent Performance All Week",
            f"Revenue is evenly distributed across weekdays (avg ${weekday_avg:,.0f}/day) and weekends (avg ${weekend_avg:,.0f}/day). "
            f"Your best performing day historically is <strong>{best_day}</strong>. "
            "Consider testing a mid-week promotion on slower days to smooth out any dips and maintain consistent cash flow."))

    # Rule 4 — Top category
    top_cat_revenue = df_raw.groupby("Category")["Total Spent"].sum()[top_category]
    total_rev       = df_raw["Total Spent"].sum()
    top_cat_share   = top_cat_revenue / total_rev * 100

    recommendations.append(("success", f"🏆 Double Down on {top_category}",
        f"<strong>{top_category}</strong> is your top revenue category, generating <strong>${top_cat_revenue:,.0f}</strong> "
        f"({top_cat_share:.1f}% of total revenue). "
        "Ensure this category is always well-stocked and prominently featured in your store or online shop. "
        "Consider expanding the product range within this category and using it as an anchor "
        "to cross-sell products from lower-performing categories."))

    # Rule 5 — Underperforming category
    bottom_cat_revenue = df_raw.groupby("Category")["Total Spent"].sum()[bottom_category]
    bottom_cat_share   = bottom_cat_revenue / total_rev * 100

    recommendations.append(("warning", f"📉 Review Strategy for {bottom_category}",
        f"<strong>{bottom_category}</strong> is the lowest revenue category with <strong>${bottom_cat_revenue:,.0f}</strong> "
        f"({bottom_cat_share:.1f}% of total revenue). "
        "Investigate whether this is due to low demand, poor visibility, or pricing issues. "
        "Consider running a targeted promotion to test demand, or evaluate whether reallocating "
        "budget to higher-performing categories would improve overall profitability."))

    # Rule 6 — Average order value
    recommendations.append(("info", "💰 Grow Average Order Value",
        f"The current average transaction value is <strong>${avg_order_value:,.0f}</strong>. "
        "Even a 10% increase in average order value would add approximately "
        f"<strong>${avg_order_value * 0.10 * total_orders / 30:,.0f}/month</strong> in additional revenue without acquiring new customers. "
        "Tactics include: product bundling, 'buy more save more' offers, "
        "free shipping thresholds, and upselling at checkout."))

    # Render cards
    for card_type, title, text in recommendations:
        st.markdown(f"""
        <div class="rec-card {card_type}">
            <div class="rec-title">{title}</div>
            <div class="rec-text">{text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("---")
    st.subheader("What Drives the Forecast — SHAP Feature Importance")
    st.markdown("Shows which features have the biggest impact on the revenue prediction model.")

    col1, col2 = st.columns(2)
    with col1:
        st.image("outputs/shap_bar.png", use_container_width=True)
    with col2:
        st.image("outputs/shap_summary.png", use_container_width=True)



    st.subheader("Forecast Data Table")
    st.markdown("Day-by-day breakdown of the 30-day forecast across all three scenarios.")

    export_df = forecast_df[["date", "forecast", "optimistic", "pessimistic"]].copy()
    export_df.columns = ["Date", "Forecast ($)", "Optimistic ($)", "Pessimistic ($)"]
    export_df = export_df.round(2)
    export_df["Date"] = pd.to_datetime(export_df["Date"]).dt.strftime("%Y-%m-%d")

    st.dataframe(export_df, use_container_width=True, hide_index=True)

    st.markdown("")
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Forecast as CSV",
        data=csv,
        file_name="forecast_30days.csv",
        mime="text/csv"
    )
