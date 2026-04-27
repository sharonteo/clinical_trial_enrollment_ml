# Clinical Trial Enrollment Dashboard

A Streamlit demo that forecasts clinical trial enrollment and identifies high-risk or zero-enrollment sites.

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

## Interview talking point

“I intentionally kept the feature set focused and operationally meaningful. The goal is not just to build a model, but to help study teams identify zero-enrollment and high-risk sites early enough to intervene.”
