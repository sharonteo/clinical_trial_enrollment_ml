import streamlit as st
import plotly.express as px

from data_generation import generate_synthetic_data
from models import build_site_risk_model, forecast_trial_enrollment


st.set_page_config(
    page_title="Clinical Trial Enrollment Dashboard",
    page_icon="🧬",
    layout="wide",
)

st.title("Clinical Trial Enrollment Forecasting Dashboard")

st.markdown("""
### Purpose
Predict trial enrollment and identify high-risk or zero-enrollment sites so clinical operations teams can take early action.
""")

# Parameters
target_enrollment = 600
forecast_months = 6

# Load data
site_df, monthly_df = generate_synthetic_data()
model, scored_sites, auc = build_site_risk_model(site_df)
forecast_df = forecast_trial_enrollment(monthly_df, forecast_months=forecast_months)

# Metrics
actual_enrolled = int(monthly_df["monthly_enrollment"].sum())
projected_enrollment = int(forecast_df["forecast_cumulative_enrollment"].max())
high_risk_sites = int((scored_sites["risk_score"] > 0.66).sum())
zero_enrollment_sites = int((scored_sites["total_enrolled"] == 0).sum())

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Current Enrollment", f"{actual_enrolled:,}")
col2.metric("Projected Enrollment", f"{projected_enrollment:,}")
col3.metric("Target", f"{target_enrollment:,}")
col4.metric("High Risk Sites", f"{high_risk_sites:,}")
col5.metric("Zero Enrollment", f"{zero_enrollment_sites:,}")

# Enrollment Progress
progress = actual_enrolled / target_enrollment

st.markdown("### Enrollment Progress")
st.progress(min(progress, 1.0))
st.write(f"{actual_enrolled:,} / {target_enrollment:,} ({progress:.0%})")

# Key Decision
st.markdown("### Key Decision")

if projected_enrollment < target_enrollment:
    st.error("Enrollment is projected to miss target. Increase support for high-risk sites or consider adding new sites.")
else:
    st.success("Enrollment is projected to meet or exceed target. Continue monitoring.")

if zero_enrollment_sites > 0:
    st.warning(f"{zero_enrollment_sites} sites have zero enrollment — these are immediate intervention candidates.")

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "Enrollment Forecast",
    "Site Risk Model",
    "Segment Review",
    "Operational Summary"
])

# =========================
# TAB 1: FORECAST
# =========================
with tab1:
    st.subheader("Enrollment Forecast")

    plot_df = forecast_df.copy()
    plot_df["target_enrollment"] = target_enrollment

    fig = px.line(
        plot_df,
        x="month",
        y=[
            "cumulative_enrollment",
            "forecast_cumulative_enrollment",
            "target_enrollment",
        ],
        markers=True,
        title="Actual vs Forecasted Enrollment"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
**How to read this:**  
This view compares actual cumulative enrollment against forecasted enrollment and the overall study target.
The goal is to identify whether the study is likely to miss the target early enough for the team to intervene.
""")

# =========================
# TAB 2: RISK MODEL
# =========================
with tab2:
    st.subheader("Site Risk Model")

    st.write(f"ROC-AUC: **{auc:.2f}**")

    st.markdown("""
The model uses a focused set of operationally meaningful variables:
historical enrollment rate, startup delay, prescreen volume, screen failure rate, prior trial count, and region.
""")

    # Top 3 High Risk
    st.markdown("### Top 3 High-Risk Sites")

    top3 = scored_sites.sort_values("risk_score", ascending=False).head(3)

    for _, row in top3.iterrows():
        st.write(
            f"**{row['site_id']}** — Risk Score: {row['risk_score']:.2f}, "
            f"Total Enrolled: {int(row['total_enrolled'])}"
        )

    # Full table
    st.markdown("### All Sites")

    display_cols = [
        "site_id",
        "region",
        "risk_score",
        "risk_band",
        "historical_enrollment_rate",
        "startup_delay_weeks",
        "prescreen_volume",
        "screen_failure_rate",
        "prior_trial_count",
        "total_enrolled",
        "zero_enroller",
    ]

    st.dataframe(
        scored_sites[display_cols].sort_values("risk_score", ascending=False),
        use_container_width=True,
    )

    # Distribution
    fig = px.histogram(scored_sites, x="risk_score", nbins=25, title="Distribution of Site Risk Scores")
    st.plotly_chart(fig, use_container_width=True)

    # Zero Enrollment Table
    st.markdown("### Zero Enrollment Sites")

    zero_df = scored_sites[scored_sites["zero_enroller"] == 1]

    if zero_df.empty:
        st.success("No zero-enrollment sites.")
    else:
        st.error(f"{len(zero_df)} zero-enrollment sites detected — immediate intervention recommended.")
        st.dataframe(
            zero_df[display_cols].sort_values("risk_score", ascending=False),
            use_container_width=True,
        )

# =========================
# TAB 3: SEGMENT
# =========================
with tab3:
    st.subheader("Segment Review")

    summary = (
        scored_sites.groupby("region", as_index=False)
        .agg(
            avg_risk_score=("risk_score", "mean"),
            avg_enrollment=("total_enrolled", "mean"),
            zero_enrollment_sites=("zero_enroller", "sum"),
            site_count=("site_id", "count"),
        )
        .sort_values("avg_risk_score", ascending=False)
    )

    st.dataframe(summary, use_container_width=True)

    fig = px.bar(
        summary,
        x="region",
        y="avg_risk_score",
        title="Average Risk Score by Region"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig_zero = px.bar(
        summary,
        x="region",
        y="zero_enrollment_sites",
        title="Zero-Enrollment Sites by Region"
    )
    st.plotly_chart(fig_zero, use_container_width=True)

# =========================
# TAB 4: SUMMARY
# =========================
with tab4:
    st.subheader("Operational Summary")

    gap = target_enrollment - projected_enrollment
    top_sites = scored_sites.sort_values("risk_score", ascending=False).head(5)
    zero_df = scored_sites[scored_sites["zero_enroller"] == 1]

    if gap > 0:
        st.write(f"Enrollment is projected to miss target by **{gap:,}** patients.")
    else:
        st.write(f"Enrollment is projected to exceed target by **{abs(gap):,}** patients.")

    st.markdown("""
### Recommended Actions
1. Prioritize zero-enrollment sites first  
2. Review high-risk sites with low prescreen volume or high screen failure rates  
3. Investigate startup delays and recruitment barriers  
4. Consider adding future sites in stronger-performing regions  
""")

    # Zero Enrollment Driver Summary
    st.markdown("### Zero Enrollment Driver Review")

    if zero_df.empty:
        st.success("No zero-enrollment sites to review.")
    else:
        driver_summary = zero_df[
            [
                "historical_enrollment_rate",
                "startup_delay_weeks",
                "prescreen_volume",
                "screen_failure_rate",
                "prior_trial_count",
            ]
        ].mean().reset_index()

        driver_summary.columns = ["Variable", "Average for Zero-Enrollment Sites"]

        st.write(
            "These averages help explain whether zero-enrollment sites appear to be driven by "
            "low historical performance, startup delays, weak prescreen volume, high screen failure, "
            "or limited prior trial experience."
        )

        st.dataframe(driver_summary, use_container_width=True)

    # Site Expansion Recommendation
    st.markdown("### Site Expansion Recommendation")

    region_perf = (
        scored_sites.groupby("region", as_index=False)
        .agg(
            avg_risk_score=("risk_score", "mean"),
            avg_enrollment=("total_enrolled", "mean"),
            zero_enrollment_sites=("zero_enroller", "sum"),
            site_count=("site_id", "count"),
        )
        .sort_values(["avg_risk_score", "avg_enrollment"], ascending=[True, False])
    )

    top_regions = region_perf.head(2)

    st.write(
        "Based on lower predicted risk, stronger enrollment performance, and fewer zero-enrollment sites, "
        "consider adding future trial sites in:"
    )

    for _, row in top_regions.iterrows():
        st.write(
            f"• **{row['region']}** — Avg Risk: {row['avg_risk_score']:.2f}, "
            f"Avg Enrollment: {row['avg_enrollment']:.1f}, "
            f"Zero-Enrollers: {int(row['zero_enrollment_sites'])}"
        )

    st.markdown("### Highest Priority Sites")

    st.dataframe(
        top_sites[
            [
                "site_id",
                "region",
                "risk_score",
                "risk_band",
                "screen_failure_rate",
                "prescreen_volume",
                "historical_enrollment_rate",
                "startup_delay_weeks",
                "total_enrolled",
                "zero_enroller",
            ]
        ],
        use_container_width=True,
    )

    with st.expander("View Raw Data and Variable Definitions"):
        st.markdown("### Site-Level Data")
        st.dataframe(site_df, use_container_width=True)

        st.download_button(
            "Download Site Data",
            site_df.to_csv(index=False),
            "site_data.csv",
            "text/csv",
        )

        st.markdown("### Monthly Enrollment Data")
        st.dataframe(monthly_df, use_container_width=True)

        st.download_button(
            "Download Monthly Data",
            monthly_df.to_csv(index=False),
            "monthly_data.csv",
            "text/csv",
        )

        st.markdown("### Variable Definitions")

        with open("data_dictionary.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())

        with open("data_dictionary.md", "rb") as f:
            st.download_button(
                "Download Data Dictionary",
                f,
                "data_dictionary.md",
                "text/markdown",
            )

st.divider()

st.caption("Future: Bayesian updating could refine predictions as new site-level data arrives.")
st.caption("Synthetic data. For demonstration purposes.")
