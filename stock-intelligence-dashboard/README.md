# Stock Intelligence Dashboard

An interactive stock analysis and forecasting dashboard built with Python.

## Features
- Historical adjusted closing prices
- 20-day and 50-day moving averages
- Interactive Plotly charts
- Prophet-based statistical forecasts and model intervals
- CSV downloads for historical and forecast data

## Setup on Windows

Open this folder in VS Code and choose **Terminal → New Terminal**.

Create and activate a virtual environment (Python 3.11 is recommended for compatibility):

```cmd
py -3.11 -m venv .venv
.venv\\Scripts\\activate
```

Install dependencies:

```cmd
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the dashboard:

```cmd
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Disclaimer
For educational purposes only. Forecasts are uncertain statistical estimates and are not investment advice.
