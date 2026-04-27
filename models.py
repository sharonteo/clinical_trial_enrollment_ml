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
