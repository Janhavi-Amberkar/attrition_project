# Workforce Attrition Patterns and Risk Hotspot Analysis | Palo Alto Networks

Diagnostic (descriptive) analysis of 1,470 employees: where attrition happens and who is most affected.

**Live dashboard:** <paste Streamlit link here>

## Key findings
- Overall attrition: 16.12% (237 of 1,470)
- Highest-rate role: Sales Representative (39.8%); largest source of exits: Laboratory Technician (62)
- First 2 years: 29.8% attrition vs 12.0% afterwards (2.49x)
- Overtime: 30.5% vs 10.4%; overtime plus frequent travel: Workload Attrition Index 260

## Run locally
    pip install -r requirements.txt
    python -m streamlit run app.py

## Files
- `app.py`: Streamlit dashboard
- `notebooks/`: cleaning and EDA notebook
- `cleaned_attrition.csv`: cleaned dataset used by the app

Findings are associations from a single snapshot, not causal claims.