import pandas as pd
import numpy as np


def generate_synthetic_data(n_sites=120, n_months=12, seed=42, n_zero_sites=3):
    """
    Generate synthetic clinical trial enrollment data.

    The data intentionally includes exactly 3 zero-enrollment sites
    because zero-enrollers are an important operational signal in clinical trial enrollment.
    """
    rng = np.random.default_rng(seed)

    regions = ["North America", "Europe", "Asia-Pacific", "Latin America"]

    # Force exactly 3 true zero-enrollment sites for demo storytelling.
    zero_site_numbers = set(rng.choice(np.arange(1, n_sites + 1), size=n_zero_sites, replace=False))

    site_rows = []
    monthly_rows = []

    for site_number in range(1, n_sites + 1):
        site_id = f"SITE_{site_number:03d}"

        region = rng.choice(regions, p=[0.45, 0.30, 0.15, 0.10])
        is_zero_enroller = site_number in zero_site_numbers

        if is_zero_enroller:
            historical_enrollment_rate = max(0, rng.normal(0.5, 0.4))
            startup_delay_weeks = max(0, rng.normal(10, 2.0))
            prescreen_volume = max(1, rng.normal(8, 4))
            screen_failure_rate = np.clip(rng.normal(0.62, 0.10), 0.20, 0.90)
            prior_trial_count = max(0, int(rng.normal(1, 1)))
            monthly_base = 0.0
        else:
            historical_enrollment_rate = max(0, rng.normal(4.0, 2.0))
            startup_delay_weeks = max(0, rng.normal(6, 2.5))
            prescreen_volume = max(5, rng.normal(60, 25))
            screen_failure_rate = np.clip(rng.normal(0.32, 0.13), 0.05, 0.75)
            prior_trial_count = max(0, int(rng.normal(8, 4)))

            site_quality = (
                0.40 * historical_enrollment_rate
                + 0.030 * prescreen_volume
                - 2.8 * screen_failure_rate
                - 0.08 * startup_delay_weeks
                + 0.10 * prior_trial_count
            )

            monthly_base = max(0.10, site_quality / 3.5)

        cumulative = 0
        site_monthly_enrollment = []

        for month in range(1, n_months + 1):
            if is_zero_enroller:
                enrolled = 0
            else:
                lam = max(0.05, monthly_base * rng.normal(1.0, 0.30))
                enrolled = rng.poisson(lam)

            site_monthly_enrollment.append(enrolled)
            cumulative += enrolled

        # Prevent accidental zero-enrollment sites outside the intentionally selected 3.
        if not is_zero_enroller and cumulative == 0:
            site_monthly_enrollment[-1] = 1
            cumulative = 1

        running_total = 0
        for month, enrolled in enumerate(site_monthly_enrollment, start=1):
            running_total += enrolled
            monthly_rows.append({
                "site_id": site_id,
                "month": month,
                "region": region,
                "monthly_enrollment": enrolled,
                "cumulative_enrollment": running_total,
            })

        site_rows.append({
            "site_id": site_id,
            "region": region,
            "historical_enrollment_rate": round(historical_enrollment_rate, 2),
            "startup_delay_weeks": round(startup_delay_weeks, 2),
            "prescreen_volume": round(prescreen_volume, 2),
            "screen_failure_rate": round(screen_failure_rate, 2),
            "prior_trial_count": prior_trial_count,
            "total_enrolled": cumulative,
            "zero_enroller": int(cumulative == 0),
            "at_risk_low_enrollment": int(cumulative < 5),
        })

    return pd.DataFrame(site_rows), pd.DataFrame(monthly_rows)
