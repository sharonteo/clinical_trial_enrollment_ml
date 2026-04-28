import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def build_site_risk_model(site_df):
    """
    Build a focused, explainable site-level risk model.

    The feature set is intentionally lean and operationally meaningful:
    - historical enrollment rate
    - startup delay
    - prescreen volume
    - screen failure rate
    - prior trial count
    - region
    """
    features = [
        "region",
        "historical_enrollment_rate",
        "startup_delay_weeks",
        "prescreen_volume",
        "screen_failure_rate",
        "prior_trial_count",
    ]

    X = site_df[features]
    y = site_df["at_risk_low_enrollment"]

    categorical_features = ["region"]
    numeric_features = [col for col in features if col not in categorical_features]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", "passthrough", numeric_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                max_depth=5,
                class_weight="balanced",
            )),
        ]
    )

    # Stratify only when both classes have enough records.
    stratify_target = y if y.nunique() == 2 and y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify_target,
    )

    model.fit(X_train, y_train)

    test_probs = model.predict_proba(X_test)[:, 1]

    if y_test.nunique() == 2:
        auc = roc_auc_score(y_test, test_probs)
    else:
        auc = 0.0

    scored_sites = site_df.copy()
    scored_sites["risk_score"] = model.predict_proba(X)[:, 1]

    scored_sites["risk_band"] = pd.cut(
        scored_sites["risk_score"],
        bins=[-0.01, 0.33, 0.66, 1.01],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    return model, scored_sites, auc


def forecast_trial_enrollment(monthly_df, forecast_months=6):
    trial_monthly = (
        monthly_df.groupby("month", as_index=False)
        .agg(monthly_enrollment=("monthly_enrollment", "sum"))
    )

    trial_monthly["cumulative_enrollment"] = trial_monthly["monthly_enrollment"].cumsum()

    X = trial_monthly[["month"]]
    y = trial_monthly["cumulative_enrollment"]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=4,
    )
    model.fit(X, y)

    future = pd.DataFrame({
        "month": range(1, int(X["month"].max()) + forecast_months + 1)
    })

    future["forecast_cumulative_enrollment"] = model.predict(future[["month"]])

    forecast_df = future.merge(
        trial_monthly,
        on="month",
        how="left",
    )

    return forecast_df


def generate_site_barrier_notes(scored_sites):
    """
    Create synthetic site-monitoring notes from structured site risk drivers.

    This gives the dashboard a lightweight NLP use case without requiring
    external clinical notes or protected health information. In a real setting,
    these notes could come from CRA comments, site monitoring logs, call notes,
    or patient recruitment updates.
    """
    notes = []

    prescreen_median = scored_sites["prescreen_volume"].median()
    failure_median = scored_sites["screen_failure_rate"].median()
    delay_median = scored_sites["startup_delay_weeks"].median()
    enrollment_median = scored_sites["historical_enrollment_rate"].median()
    prior_trial_median = scored_sites["prior_trial_count"].median()

    for _, row in scored_sites.iterrows():
        barriers = []

        if row["prescreen_volume"] < prescreen_median:
            barriers.append("low patient prescreen volume")
        if row["screen_failure_rate"] > failure_median:
            barriers.append("high screen failure rate")
        if row["startup_delay_weeks"] > delay_median:
            barriers.append("startup delay")
        if row["historical_enrollment_rate"] < enrollment_median:
            barriers.append("weak historical enrollment performance")
        if row["prior_trial_count"] < prior_trial_median:
            barriers.append("limited prior trial experience")

        if not barriers:
            barriers.append("stable recruitment performance")

        note = (
            f"Site {row['site_id']} in {row['region']} shows {', '.join(barriers)}. "
            f"Risk band is {row['risk_band']} with a risk score of {row['risk_score']:.2f}."
        )
        notes.append(note)

    notes_df = scored_sites[["site_id", "region", "risk_score", "risk_band"]].copy()
    notes_df["site_monitoring_note"] = notes
    return notes_df


def analyze_site_barrier_themes(notes_df):
    """
    Simple NLP-style keyword theme extraction for site barrier notes.

    This intentionally uses transparent keyword matching instead of a heavy NLP
    package so the app stays easy to run in Streamlit.
    """
    theme_keywords = {
        "Low prescreen volume": ["low patient prescreen volume"],
        "High screen failure": ["high screen failure rate"],
        "Startup delay": ["startup delay"],
        "Weak historical enrollment": ["weak historical enrollment performance"],
        "Limited prior trial experience": ["limited prior trial experience"],
        "Stable performance": ["stable recruitment performance"],
    }

    rows = []
    for theme, keywords in theme_keywords.items():
        mask = notes_df["site_monitoring_note"].str.lower().apply(lambda text: any(keyword in text for keyword in keywords))
        matching = notes_df[mask]
        rows.append({
            "theme": theme,
            "site_count": int(mask.sum()),
            "avg_risk_score": float(matching["risk_score"].mean()) if not matching.empty else 0.0,
            "example_sites": ", ".join(matching["site_id"].head(3).astype(str).tolist()),
        })

    return pd.DataFrame(rows).sort_values(["site_count", "avg_risk_score"], ascending=[False, False])


def generate_llm_operational_summary(scored_sites, forecast_df, target_enrollment):
    """
    Generate an LLM-style executive narrative from model outputs.

    This is a deterministic stand-in for an LLM call. It demonstrates how an LLM
    layer could convert ML results into an operational summary for clinical teams.
    """
    actual_enrollment = int(forecast_df["cumulative_enrollment"].dropna().max())
    projected_enrollment = int(forecast_df["forecast_cumulative_enrollment"].max())
    gap = target_enrollment - projected_enrollment
    high_risk_count = int((scored_sites["risk_score"] > 0.66).sum())
    zero_count = int((scored_sites["zero_enroller"] == 1).sum())
    top_site = scored_sites.sort_values("risk_score", ascending=False).iloc[0]
    action = "increase recruitment support, review screening criteria, and consider adding sites in stronger regions" if gap > 0 else "continue monitoring high-risk sites while protecting the current enrollment trajectory"

    return f"""
**AI-generated operational readout**

Enrollment currently stands at **{actual_enrollment:,}** patients and is projected to reach **{projected_enrollment:,}** against a target of **{target_enrollment:,}**. The forecasted gap is **{max(gap, 0):,}** patients.

The model identified **{high_risk_count}** high-risk sites and **{zero_count}** zero-enrollment sites. The highest-priority site is **{top_site['site_id']}** in **{top_site['region']}**, with a risk score of **{top_site['risk_score']:.2f}**.

Recommended next step: **{action}**.
""".strip()
