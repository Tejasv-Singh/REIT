#!/usr/bin/env python
"""
Setup checker script to diagnose issues before running Streamlit app.
Run this before running streamlit run app.py
"""

import sys
from pathlib import Path

print("=" * 60)
print("REIT/InvIT Quantitative Analysis - Setup Checker")
print("=" * 60)

errors = []
warnings = []

# Check Python version
print("\n1. Checking Python version...")
if sys.version_info < (3, 8):
    errors.append(f"Python 3.8+ required. Found: {sys.version}")
    print(f"   ✗ Python version: {sys.version}")
else:
    print(f"   ✓ Python version: {sys.version_info.major}.{sys.version_info.minor}")

# Check required packages
print("\n2. Checking required packages...")
required_packages = [
    'streamlit', 'pandas', 'numpy', 'scipy', 'plotly', 
    'pydantic', 'cvxpy', 'statsmodels'
]

for package in required_packages:
    try:
        __import__(package)
        print(f"   ✓ {package} installed")
    except ImportError:
        errors.append(f"Missing package: {package}")
        print(f"   ✗ {package} NOT installed")

# Check project structure
print("\n3. Checking project structure...")
required_files = [
    'app.py',
    'requirements.txt',
    'data/sample_assets.csv',
    'models/__init__.py',
    'models/stochastic_processes.py',
    'models/cashflow_simulator.py',
    'models/risk_metrics.py',
    'models/portfolio_optimization.py',
    'utils/__init__.py',
    'utils/data_loader.py',
    'utils/plotting.py',
    'utils/config.py',
]

for file_path in required_files:
    path = Path(file_path)
    if path.exists():
        print(f"   ✓ {file_path}")
    else:
        errors.append(f"Missing file: {file_path}")
        print(f"   ✗ {file_path} NOT found")

# Check data file
print("\n4. Checking data file...")
data_path = Path("data/sample_assets.csv")
if data_path.exists():
    print(f"   ✓ Data file found: {data_path.absolute()}")
    # Try to read it
    try:
        import pandas as pd
        df = pd.read_csv(data_path)
        print(f"   ✓ Data file readable ({len(df)} rows)")
    except Exception as e:
        warnings.append(f"Data file exists but cannot be read: {e}")
        print(f"   ⚠ Data file exists but cannot be read: {e}")
else:
    errors.append("Data file not found")
    print(f"   ✗ Data file NOT found")

# Check imports
print("\n5. Testing imports...")
try:
    sys.path.insert(0, str(Path.cwd()))
    from utils.data_loader import load_assets
    print("   ✓ utils.data_loader")
except Exception as e:
    errors.append(f"Import error (utils.data_loader): {e}")
    print(f"   ✗ utils.data_loader: {e}")

try:
    from models.cashflow_simulator import CashflowSimulator
    print("   ✓ models.cashflow_simulator")
except Exception as e:
    errors.append(f"Import error (models.cashflow_simulator): {e}")
    print(f"   ✗ models.cashflow_simulator: {e}")

try:
    from models.risk_metrics import compute_var
    print("   ✓ models.risk_metrics")
except Exception as e:
    errors.append(f"Import error (models.risk_metrics): {e}")
    print(f"   ✗ models.risk_metrics: {e}")

try:
    from models.portfolio_optimization import mean_variance_optimization
    print("   ✓ models.portfolio_optimization")
except Exception as e:
    errors.append(f"Import error (models.portfolio_optimization): {e}")
    print(f"   ✗ models.portfolio_optimization: {e}")

try:
    from utils.plotting import plot_sample_paths
    print("   ✓ utils.plotting")
except Exception as e:
    errors.append(f"Import error (utils.plotting): {e}")
    print(f"   ✗ utils.plotting: {e}")

try:
    from utils.config import DEFAULT_SIMULATION_PARAMS
    print("   ✓ utils.config")
except Exception as e:
    errors.append(f"Import error (utils.config): {e}")
    print(f"   ✗ utils.config: {e}")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

if errors:
    print(f"\n❌ Found {len(errors)} error(s):")
    for i, error in enumerate(errors, 1):
        print(f"   {i}. {error}")
    print("\n⚠️  Please fix these errors before running the Streamlit app.")
    print("\nTo install missing packages, run:")
    print("   pip install -r requirements.txt")
    sys.exit(1)
elif warnings:
    print(f"\n⚠️  Found {len(warnings)} warning(s):")
    for i, warning in enumerate(warnings, 1):
        print(f"   {i}. {warning}")
    print("\n✓ Setup looks good, but check warnings above.")
else:
    print("\n✅ All checks passed! You can run the Streamlit app:")
    print("   streamlit run app.py")

print("=" * 60)

