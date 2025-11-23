# Troubleshooting Guide

## Common Errors and Solutions

### Error 1: ModuleNotFoundError (pandas, numpy, streamlit, etc.)

**Error Message:**
```
ModuleNotFoundError: No module named 'pandas'
```

**Solution:**
Install all required dependencies:
```bash
pip install -r requirements.txt
```

If you're using a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

### Error 2: Import Error - Cannot import from utils or models

**Error Message:**
```
ImportError: cannot import name 'load_assets' from 'utils.data_loader'
```

**Solution:**
1. Make sure you're running from the project root directory:
   ```bash
   cd /home/hada_linux/Projects/ML/REIT
   streamlit run app.py
   ```

2. Check that all `__init__.py` files exist:
   - `models/__init__.py`
   - `utils/__init__.py`
   - `tests/__init__.py`

---

### Error 3: FileNotFoundError - sample_assets.csv not found

**Error Message:**
```
FileNotFoundError: Could not find sample_assets.csv
```

**Solution:**
1. Verify the file exists:
   ```bash
   ls data/sample_assets.csv
   ```

2. If missing, the file should be at: `data/sample_assets.csv`

3. Make sure you're running Streamlit from the project root directory

---

### Error 4: Import Error in cashflow_simulator.py

**Error Message:**
```
ImportError: attempted relative import with no known parent package
```

**Solution:**
This has been fixed in the code. The `cashflow_simulator.py` now handles both relative and absolute imports. If you still see this error:

1. Make sure you're running `streamlit run app.py` (not `python app.py`)
2. The import fallback should handle this automatically

---

### Error 5: Optimization Fails

**Error Message:**
```
ValueError: Optimization failed with status: ...
```

**Solution:**
1. Make sure at least 2 assets are selected
2. Ensure simulations have completed successfully
3. Try adjusting the target return value
4. Check that cvxpy or scipy is installed

---

### Error 6: Streamlit Command Not Found

**Error Message:**
```
streamlit: command not found
```

**Solution:**
```bash
pip install streamlit
```

Or if using a virtual environment:
```bash
source venv/bin/activate
pip install streamlit
```

---

## Quick Diagnostic

Run the setup checker before running Streamlit:

```bash
python check_setup.py
```

This will check:
- Python version
- Required packages
- Project structure
- Data files
- Import functionality

---

## Step-by-Step Setup (If Starting Fresh)

1. **Navigate to project directory:**
   ```bash
   cd /home/hada_linux/Projects/ML/REIT
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify setup:**
   ```bash
   python check_setup.py
   ```

5. **Run the app:**
   ```bash
   streamlit run app.py
   ```

---

## Still Having Issues?

1. Check Python version: `python --version` (should be 3.8+)
2. Verify you're in the correct directory
3. Ensure virtual environment is activated
4. Run `python check_setup.py` for detailed diagnostics
5. Check that all files from the project structure exist

---

## Common Streamlit-Specific Issues

### Issue: App reloads constantly

**Solution:** This is normal Streamlit behavior when code changes. If it's excessive, check for:
- Infinite loops in code
- Large data processing without caching
- File watchers triggering reloads

### Issue: Charts not displaying

**Solution:**
1. Check browser console for JavaScript errors
2. Verify plotly is installed: `pip install plotly`
3. Try a different browser

### Issue: Slow performance

**Solution:**
1. Reduce number of simulations (n_sims)
2. Reduce time horizon
3. Use fewer time steps per year
4. The `@st.cache_data` decorator should help with caching

---

## Getting Help

If you continue to experience issues:
1. Run `python check_setup.py` and share the output
2. Check the error message carefully
3. Verify all steps in the setup instructions
4. Ensure you're using Python 3.8 or higher

