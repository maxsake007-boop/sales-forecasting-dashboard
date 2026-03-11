# Sales Forecasting Dashboard

An interactive Streamlit application that forecasts retail revenue 
across three scenarios — optimistic, base, and pessimistic — 
helping businesses plan inventory, staffing, and budgets with confidence.

## Demo
> Add screenshots here

## Features
- 30-day revenue forecast with 3 scenarios
- SHAP analysis — understand which factors drive the forecast
- Interactive charts and KPI cards
- EDA with 10+ visualizations
- Clean and intuitive interface

## Tech Stack
- **ML Model:** LightGBM
- **Dashboard:** Streamlit, Plotly
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Visualization:** Matplotlib, Seaborn, SHAP

## Project Structure
```
ПРОЕКТ 1/
├── app/
│   └── main.py              # Streamlit dashboard
├── data/
│   ├── raw/                 # Original dataset
│   └── processed/           # Cleaned and featured data
├── models/
│   └── forecast_model_v1.pkl
├── outputs/
│   ├── PNG/                 # EDA visualizations
│   ├── forecast_30days.csv
│   ├── shap_bar.png
│   └── shap_summary.png
├── src/
│   ├── cleaning.py          # Data cleaning functions
│   └── model.py             # Model functions
├── scripts/
│   ├── run_cleaning.py      # Run data cleaning
│   ├── run_eda.py           # Run EDA
│   ├── run_features.py      # Run feature engineering
│   └── run_model.py         # Train model
├── requirements.txt
└── .gitignore
```

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/sales-forecasting-dashboard.git
cd sales-forecasting-dashboard
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the dashboard
```bash
streamlit run app/main.py
```

## Model Performance
- **Algorithm:** LightGBM
- **MAPE:** 28.9%
- **Key features:** lag_14, rolling_mean_14

## Author
Max Narbekov — Data Analyst / Data Scientist
```

Теперь `.gitignore` — у тебя он уже есть, но провери что там есть эти строки:
```
__pycache__/
*.pyc
.venv/
venv/
.idea/
.env
data/raw/