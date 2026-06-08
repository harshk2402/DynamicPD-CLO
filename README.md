# DynamicPD-CLO

Dynamic probability of default estimation for CLO tranche early warning using gradient boosting.

## What it does

Estimates quarterly PD for leveraged loan borrowers using XGBoost/LightGBM over firm financials, market data, and macro variables. Feeds dynamic PD estimates into CLO tranche pricing via Gaussian and Student-t copula Monte Carlo simulation. Evaluates whether ML-estimated PDs provide earlier warning of tranche stress than static rating agency assessments.

## Data sources

WRDS (Compustat, CRSP, Capital IQ, DealScan, TRACE, IBES, CBOE VIX), FMP, FRED.

## Setup

1. Copy `.env.example` to `.env` and fill in credentials.
2. Create and activate the virtual environment: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`

## License

MIT License — Copyright (c) 2026 Harsh Kulkarni
