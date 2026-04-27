
# Clinical Trial Enrollment Forecasting Dashboard

A Streamlit demo that forecasts clinical trial enrollment and identifies high-risk or zero-enrollment sites.

https://trialenrollment.streamlit.app/

## Why this version is focused

The project uses a lean, explainable feature set instead of a large number of variables. This keeps the demo aligned with clinical operations decision-making:

- Which sites are underperforming?
- Which sites have zero enrollment?
- What risk drivers may explain the issue?
- Where should the study team intervene?

## Files

- `app.py` — Streamlit dashboard
- `data_generation.py` — Synthetic trial/site data generation
- `models.py` — Site risk model and enrollment forecast model
- `data_dictionary.md` — Variable definitions
- `requirements.txt` — Python dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```



