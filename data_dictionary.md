# Data Dictionary

This project uses a focused, explainable feature set for a synthetic clinical trial enrollment dashboard.

## Site-Level Fields

| Field | Description |
|---|---|
| site_id | Synthetic research site identifier |
| region | Geographic region where the site operates |
| historical_enrollment_rate | Prior enrollment performance proxy |
| startup_delay_weeks | Estimated delay before a site becomes operational |
| prescreen_volume | Number of patients prescreened |
| screen_failure_rate | Share of prescreened patients who fail eligibility criteria |
| prior_trial_count | Prior trial participation experience |
| total_enrolled | Total patients enrolled by the site over the observed period |
| zero_enroller | Indicates whether the site enrolled zero patients |
| at_risk_low_enrollment | Target label for low-enrollment risk |
| risk_score | Predicted probability that a site is at risk for low enrollment |
| risk_band | Low, Medium, or High risk category based on risk score |

## Monthly Enrollment Fields

| Field | Description |
|---|---|
| site_id | Synthetic research site identifier |
| month | Study month |
| region | Geographic region |
| monthly_enrollment | Patients enrolled during the month |
| cumulative_enrollment | Running enrollment count for the site |

## Modeling Notes

The model intentionally uses a lean feature set because clinical operations teams need risk drivers that are easy to interpret and act on.  
Zero-enrollment sites are intentionally included in the synthetic data because they represent a common operational issue in clinical trial enrollment.
