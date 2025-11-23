# REIT/InvIT Quantitative Cashflow & NAV Simulation Tool

A production-quality quantitative finance tool for simulating future cashflows and NAV (Net Asset Value) of REIT (Real Estate Investment Trust) and InvIT (Infrastructure Investment Trust) assets using stochastic processes and Monte Carlo methods. The tool includes portfolio optimization capabilities and an interactive Streamlit web dashboard.

## 🎯 Features

- **Stochastic Cashflow Simulation**: Models rent/tariff growth, occupancy/utilization, and infrastructure shocks
- **NAV Evolution**: Simulates Net Asset Value over time based on cashflows and distributions
- **Risk Metrics**: Computes VaR, CVaR, Sharpe ratio, and downside risk
- **Portfolio Optimization**: Markowitz mean-variance optimization with efficient frontier
- **Interactive Dashboard**: Streamlit web interface with multiple analysis tabs
- **Monte Carlo Methods**: Thousands of simulation paths for robust statistical analysis

## 📦 Tech Stack

- **Python 3.10+**
- **Core Libraries**: `pandas`, `numpy`, `scipy`, `statsmodels`
- **Visualization**: `matplotlib`, `plotly`
- **UI**: `streamlit`
- **Validation**: `pydantic`
- **Optimization**: `cvxpy` (with `scipy.optimize` fallback)

## 🚀 Setup Instructions

### 1. Clone or Navigate to Project Directory

```bash
cd /path/to/REIT
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**On Linux/Mac:**
```bash
source venv/bin/activate
```

**On Windows:**
```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`.

## 📁 Project Structure

```
reit_invit_quant_project/
├── app.py                          # Streamlit entry point
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── data/
│   └── sample_assets.csv           # Example REIT/InvIT asset data
├── models/
│   ├── __init__.py
│   ├── stochastic_processes.py     # GBM, OU, Poisson jumps
│   ├── cashflow_simulator.py       # Cashflow + NAV simulation
│   ├── portfolio_optimization.py   # Markowitz + efficient frontier
│   └── risk_metrics.py             # VaR, CVaR, downside risk
├── utils/
│   ├── __init__.py
│   ├── data_loader.py              # CSV loading & validation
│   ├── plotting.py                 # Plotly visualization functions
│   └── config.py                   # Configuration constants
├── notebooks/
│   └── exploration.ipynb           # Optional experiments
└── tests/
    ├── test_stochastic.py
    ├── test_cashflow_sim.py
    └── test_portfolio_opt.py
```

## 📊 Methodology

### Cashflow Simulation

The simulation engine models the following components:

1. **Revenue Calculation**:
   - Rent/tariff growth: Modeled using **Geometric Brownian Motion (GBM)**
     - `dS_t = μ S_t dt + σ S_t dW_t`
   - Occupancy/utilization: Modeled using **Ornstein-Uhlenbeck (OU) process**
     - `dX_t = κ(θ - X_t)dt + σ dW_t`
     - Bounded between 0 and 1, mean-reverting to target occupancy
   - Revenue = Rent/Tariff × Occupancy × Scale Factor

2. **Operating Income**:
   - Operating Expenses = Revenue × Operating Expense Ratio
   - Net Operating Income (NOI) = Revenue - Operating Expenses

3. **Debt Service**:
   - Interest Expense = Debt Ratio × Initial NAV × Debt Cost (annualized)

4. **Distributable Cashflow (DCF)**:
   - DCF = max(0, NOI - Interest Expense)
   - Distribution = DCF × Payout Ratio

5. **NAV Evolution**:
   - NAV_{t+1} = NAV_t + (DCF_t - Distribution_t) - Capex_t
   - NAV cannot go negative (liquidation scenario)

### Risk Metrics

- **Value-at-Risk (VaR)**: Maximum loss expected at a given confidence level (e.g., 95%)
- **Conditional VaR (CVaR)**: Expected loss given that loss exceeds VaR (tail risk)
- **Downside Risk**: Standard deviation of returns below Minimum Acceptable Return (MAR)
- **Sharpe Ratio**: Risk-adjusted return = (Return - Risk-free Rate) / Volatility

### Portfolio Optimization

The tool implements **Markowitz Mean-Variance Optimization**:

- **Objective**: Minimize portfolio variance (risk)
- **Constraints**:
  - Portfolio weights sum to 1.0 (fully invested)
  - No short selling (weights ≥ 0)
  - Optional: Target return constraint

- **Efficient Frontier**: Set of optimal portfolios that maximize return for a given level of risk

The optimization uses `cvxpy` (preferred) or `scipy.optimize` as a fallback.

## 🖥️ Usage Guide

### Dashboard Tabs

#### 1. Overview Tab
- Summary table of all selected assets
- Key parameters display (growth rates, payout ratios, debt ratios, etc.)
- Portfolio-level summary statistics

#### 2. Single Asset Analysis Tab
- **Asset Selection**: Dropdown to choose one asset for detailed analysis
- **Visualizations**:
  - Sample NAV evolution paths with confidence bands
  - Distribution/cashflow paths over time
  - Ending NAV distribution histogram
- **Risk Metrics Table**: VaR, CVaR, downside risk, Sharpe ratio
- **Summary Statistics**: Annualized return, average yield, IRR approximation

#### 3. Portfolio Simulation Tab
- **Portfolio NAV Evolution**: Combined equal-weighted portfolio paths
- **Asset Cashflow Comparison**: Mean distribution paths across assets
- **Portfolio Risk Metrics**: VaR, CVaR, downside risk, Sharpe ratio at portfolio level
- **Portfolio Summary**: Average annual distributions

#### 4. Portfolio Optimization Tab
- **Efficient Frontier Plot**: Risk-return tradeoff curve
- **Optimal Portfolio Weights**: Bar chart showing recommended allocation
- **Optimal Portfolio Metrics**: Expected return, risk, Sharpe ratio, VaR/CVaR
- **Detailed Weights Table**: Asset-by-asset breakdown

### Sidebar Controls

- **Simulation Parameters**:
  - Horizon (years): 1-20 years (default: 10)
  - Steps per year: Annual, Quarterly, or Monthly (default: Monthly)
  - Number of simulations: 100-10,000 (default: 1,000)
  - VaR confidence level: 0.90-0.99 (default: 0.95)
  - Risk-free rate: Annual rate for Sharpe ratio calculation

- **Asset Selection**: Multiselect widget to choose assets for analysis

- **Portfolio Options**:
  - Enable Portfolio Optimization checkbox
  - Target return (optional): If specified, optimizes for that return; otherwise maximizes Sharpe ratio

## 🧪 Running Tests

The project includes comprehensive unit tests. Run them with:

```bash
pytest tests/
```

Or run specific test files:

```bash
pytest tests/test_stochastic.py
pytest tests/test_cashflow_sim.py
pytest tests/test_portfolio_opt.py
```

### Test Coverage

- **Stochastic Processes**: Tests for GBM (including zero-volatility case), OU (mean reversion, bounds), Poisson jumps
- **Cashflow Simulation**: Output shapes, non-negative distributions/NAV, summary stats structure
- **Portfolio Optimization**: Weight constraints (sum to 1, non-negative), covariance matrix properties, efficient frontier generation

## 📝 Data Format

The `data/sample_assets.csv` file contains the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `asset_id` | Unique identifier | REIT001 |
| `name` | Asset name | Prime Office Tower |
| `asset_type` | Type of asset | Office REIT, Road InvIT |
| `base_rent_or_tariff` | Starting rent/tariff value | 150.0 |
| `current_occupancy_or_utilization` | Current occupancy % (0-1) | 0.85 |
| `rent_tariff_growth_mu` | Expected annual growth rate | 0.04 |
| `rent_tariff_growth_sigma` | Growth volatility | 0.12 |
| `operating_expense_ratio` | Operating expenses as % of revenue | 0.25 |
| `debt_ratio` | Debt as % of NAV | 0.45 |
| `debt_cost` | Annual interest rate on debt | 0.055 |
| `payout_ratio` | Distribution payout % of DCF | 0.90 |
| `current_nav` | Current Net Asset Value | 500000000 |
| `beta` | Beta vs REIT/Infrastructure index | 0.85 |

To use your own data, replace `data/sample_assets.csv` with your file following the same format.

## 🔧 Configuration

Default simulation parameters can be modified in `utils/config.py`:

```python
DEFAULT_SIMULATION_PARAMS = {
    'horizon_years': 10,
    'steps_per_year': 12,  # Monthly
    'n_sims': 1000,
    'var_confidence': 0.95,
    'risk_free_rate': 0.03,
    'ou_kappa': 2.0,  # Mean reversion speed
    'ou_theta': 0.90,  # Long-term occupancy target
    'ou_sigma': 0.1,  # Occupancy volatility
    'capex_ratio': 0.02,  # 2% of NAV per year
}
```

## 🎓 Educational Value

This project is designed to be:

- **Student-Friendly**: Well-commented code suitable for 2nd-year engineering students
- **Production-Ready**: Proper error handling, input validation, modular design
- **Extensible**: Easy to add new stochastic processes, risk metrics, or optimization methods
- **Best Practices**: Follows DRY principles, type hints, comprehensive docstrings

## 📚 Key Concepts Explained

### Geometric Brownian Motion (GBM)
Used to model rent/tariff growth. GBM ensures:
- Values remain positive (important for prices/rents)
- Log-returns are normally distributed
- Captures exponential growth with volatility

### Ornstein-Uhlenbeck Process
Used to model occupancy/utilization. OU process:
- Mean-reverts to a long-term level (theta)
- Bounded (important for occupancy rates 0-100%)
- Captures cyclical behavior in real estate markets

### Monte Carlo Simulation
- Generates thousands of possible future scenarios
- Provides statistical distributions of outcomes
- Enables calculation of risk metrics (VaR, CVaR)

### Markowitz Optimization
- Finds optimal asset allocation
- Balances risk and return
- Efficient frontier shows all optimal portfolios

## 🐛 Troubleshooting

### Import Errors
If you encounter import errors, ensure:
1. Virtual environment is activated
2. All dependencies are installed: `pip install -r requirements.txt`
3. You're running from the project root directory

### Optimization Fails
If portfolio optimization fails:
- Ensure at least 2 assets are selected
- Check that simulation has completed successfully
- Try adjusting the target return value

### Simulation Takes Too Long
- Reduce number of simulations (n_sims)
- Reduce time horizon
- Use fewer time steps per year

## 📄 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

To extend this project:
1. Add new stochastic processes in `models/stochastic_processes.py`
2. Add new risk metrics in `models/risk_metrics.py`
3. Add new optimization methods in `models/portfolio_optimization.py`
4. Add new visualizations in `utils/plotting.py`

## 📧 Support

For questions or issues, please refer to the code documentation (docstrings) or create an issue in the project repository.

---

**Built with ❤️ for quantitative finance education and analysis**

