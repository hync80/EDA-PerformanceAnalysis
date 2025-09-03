# ===== Retail Weekly Sales Forecasting: Complete Pipeline =====
# Saves plots to ./figs and an HTML report in the working folder.
# Optional libs: prophet, statsmodels. Script runs even if either is missing.

import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
from datetime import datetime
import base64
import numpy as np
import pandas as pd

# --- Matplotlib (headless: saves images even without GUI) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------- CONFIG ---------------------------
# Change this if your CSV is in a different path
CSV_PATH = r"Walmart.csv"   # e.g., r"C:\Users\you\Downloads\Walmart.csv"
STORE_ID = 1
TEST_WEEKS = 12

# Paths relative to this script file
BASE_DIR = Path(__file__).parent
FIG_DIR = BASE_DIR / "figs"
FIG_DIR.mkdir(exist_ok=True)
REPORT_NAME = BASE_DIR / f"Retail_Forecast_Report_store{STORE_ID}.html"

# --------------------------- HELPERS ---------------------------
def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true), 1e-9, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def save_plot(fig, path: Path):
    fig.savefig(path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"[PLOT] Saved {path.resolve()}")

def png_to_base64(path: Path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# --------------------------- 1) LOAD ---------------------------
df = pd.read_csv(CSV_PATH)
# Parse dd-mm-YYYY (e.g., '05-02-2010')
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

required_cols = {"Store", "Date", "Weekly_Sales"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# --------------------------- 2) SCOPE --------------------------
store_df = df[df["Store"] == STORE_ID].copy()
if store_df.empty:
    raise ValueError(f"No rows found for Store={STORE_ID} in {CSV_PATH}.")

store_df = store_df.sort_values("Date")
ts = store_df[["Date", "Weekly_Sales"]].set_index("Date").asfreq("W-FRI")
# Fill potential gaps from asfreq
ts["Weekly_Sales"] = ts["Weekly_Sales"].interpolate(method="time")
ts.index.name = "Date"

# --------------------------- 3) SPLIT --------------------------
if len(ts) <= TEST_WEEKS + 10:
    raise ValueError("Time series too short for a 12-week test split. Reduce TEST_WEEKS or add data.")

train = ts.iloc[:-TEST_WEEKS].copy()
test  = ts.iloc[-TEST_WEEKS:].copy()

print(f"Train range: {train.index.min().date()} → {train.index.max().date()}")
print(f"Test  range:  {test.index.min().date()}  → {test.index.max().date()}")

# --------------------------- 4) MODELS -------------------------
results = []           # to collect metrics rows
forecast_frames = {}   # model_name -> DataFrame(Date, Forecast, Lower, Upper)

# ---- (A) Prophet (if available) ----
use_prophet = True
try:
    from prophet import Prophet
except Exception:
    print("Prophet not available. Skipping Prophet.")
    use_prophet = False

if use_prophet:
    p_train = train.reset_index().rename(columns={"Date": "ds", "Weekly_Sales": "y"})
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False)
    m.fit(p_train)

    future = m.make_future_dataframe(periods=TEST_WEEKS, freq="W-FRI")
    p_fcst = m.predict(future).set_index("ds")

    # Predictions aligned to test
    p_pred = p_fcst.loc[test.index, "yhat"].rename("Forecast")

    p_mape = mape(test["Weekly_Sales"], p_pred)
    p_rmse = rmse(test["Weekly_Sales"], p_pred)
    print(f"[Prophet] MAPE: {p_mape:.2f}%  RMSE: {p_rmse:,.0f}")

    prophet_out = pd.DataFrame({
        "Date": test.index,
        "Forecast": p_pred.values,
        "Lower": p_fcst.loc[test.index, "yhat_lower"].values,
        "Upper": p_fcst.loc[test.index, "yhat_upper"].values
    })
    prophet_csv = BASE_DIR / f"prophet_forecast_store{STORE_ID}.csv"
    prophet_out.to_csv(prophet_csv, index=False)
    forecast_frames["Prophet"] = prophet_out
    results.append({"Model": "Prophet", "MAPE (%)": p_mape, "RMSE": p_rmse, "CSV": str(prophet_csv.resolve())})

    # --- Plots right after Prophet so figs are always created ---
    # History plot (train + test)
    fig1 = plt.figure(figsize=(10, 4.8))
    plt.plot(train.index, train["Weekly_Sales"], label="Train")
    plt.plot(test.index, test["Weekly_Sales"], label="Test", linewidth=2)
    plt.title(f"Weekly Sales — Store {STORE_ID} (History)")
    plt.xlabel("Week"); plt.ylabel("Weekly_Sales")
    plt.legend(); plt.tight_layout()
    plot_history_path = FIG_DIR / f"history_store{STORE_ID}.png"
    save_plot(fig1, plot_history_path)

    # Actual vs Prophet forecast
    fig2 = plt.figure(figsize=(10, 4.8))
    plt.plot(test.index, test["Weekly_Sales"], label="Actual", linewidth=2)
    plt.plot(test.index, p_pred.values, label="Prophet Forecast")
    plt.title(f"Actual vs Forecast — Store {STORE_ID} (Prophet)")
    plt.xlabel("Week"); plt.ylabel("Weekly_Sales")
    plt.legend(); plt.tight_layout()
    plot_compare_path = FIG_DIR / f"actual_vs_forecast_prophet_store{STORE_ID}.png"
    save_plot(fig2, plot_compare_path)
else:
    # Even if Prophet isn’t available, still save a history plot
    fig1 = plt.figure(figsize=(10, 4.8))
    plt.plot(train.index, train["Weekly_Sales"], label="Train")
    plt.plot(test.index, test["Weekly_Sales"], label="Test", linewidth=2)
    plt.title(f"Weekly Sales — Store {STORE_ID} (History)")
    plt.xlabel("Week"); plt.ylabel("Weekly_Sales")
    plt.legend(); plt.tight_layout()
    plot_history_path = FIG_DIR / f"history_store{STORE_ID}.png"
    save_plot(fig1, plot_history_path)
    # No forecast comparison possible without a model

# ---- (B) SARIMAX (if available) ----
use_sarimax = True
try:
    import itertools
    import statsmodels.api as sm
except Exception:
    print("statsmodels not available. Skipping SARIMAX.")
    use_sarimax = False

if use_sarimax:
    # keep grid tiny for speed; expand later if needed
    p = d = q = range(0, 2)          # (0 or 1)
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(0,1,1,52)]       # one good seasonal option for weekly data
    y_train = train["Weekly_Sales"]

    best_aic = np.inf
    best_params = None
    for order in pdq:
        for sorder in seasonal_pdq:
            try:
                model = sm.tsa.statespace.SARIMAX(
                    y_train, order=order, seasonal_order=sorder,
                    enforce_stationarity=False, enforce_invertibility=False
                )
                res = model.fit(disp=False)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_params = (order, sorder)
            except Exception:
                continue

    if best_params is None:
        best_params = ((1,1,1), (0,1,1,52))
    print(f"[SARIMAX] Best params: {best_params}, AIC={best_aic:.1f}")

    order, sorder = best_params
    final_model = sm.tsa.statespace.SARIMAX(
        y_train, order=order, seasonal_order=sorder,
        enforce_stationarity=False, enforce_invertibility=False
    ).fit(disp=False)

    sarimax_pred = final_model.get_forecast(steps=TEST_WEEKS)
    sarimax_mean = sarimax_pred.predicted_mean.rename("Forecast")
    sarimax_ci = sarimax_pred.conf_int()

    lower_col = [c for c in sarimax_ci.columns if "lower" in c.lower()][0]
    upper_col = [c for c in sarimax_ci.columns if "upper" in c.lower()][0]

    s_mape = mape(test["Weekly_Sales"], sarimax_mean.values)
    s_rmse = rmse(test["Weekly_Sales"], sarimax_mean.values)
    print(f"[SARIMAX] MAPE: {s_mape:.2f}%  RMSE: {s_rmse:,.0f}")

    sarimax_out = pd.DataFrame({
        "Date": test.index,
        "Forecast": sarimax_mean.values,
        "Lower": sarimax_ci[lower_col].values,
        "Upper": sarimax_ci[upper_col].values
    })
    sarimax_csv = BASE_DIR / f"sarimax_forecast_store{STORE_ID}.csv"
    sarimax_out.to_csv(sarimax_csv, index=False)
    forecast_frames["SARIMAX"] = sarimax_out
    results.append({"Model": "SARIMAX", "MAPE (%)": s_mape, "RMSE": s_rmse, "CSV": str(sarimax_csv.resolve())})

# --------------------------- 5) HTML REPORT -------------------------
# Build metrics table
if results:
    metrics_df = pd.DataFrame(results).sort_values("MAPE (%)")
    metrics_html = metrics_df.to_html(
        index=False,
        float_format=lambda x: f"{x:,.2f}" if isinstance(x, (int,float)) else x
    )
else:
    metrics_html = "<p><b>No models ran.</b> Install prophet and/or statsmodels to generate forecasts.</p>"

# Images (history is guaranteed; prophet comparison only if prophet ran)
img_history_b64 = png_to_base64(plot_history_path)
img_compare_block = ""
compare_img_path = FIG_DIR / f"actual_vs_forecast_prophet_store{STORE_ID}.png"
if compare_img_path.exists():
    img_compare_b64 = png_to_base64(compare_img_path)
    img_compare_block = f"""
    <h2>Actual vs Forecast (Prophet)</h2>
    <div class="card">
      <img alt="comparison" src="data:image/png;base64,{img_compare_b64}" style="max-width:100%; height:auto;"/>
    </div>
    """

report_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Retail Sales Forecast — Store {STORE_ID}</title>
<style>
 body {{ font-family: Arial, sans-serif; margin: 20px; }}
 h1, h2 {{ color: #222; }}
 .card {{ border: 1px solid #ddd; padding: 16px; border-radius: 8px; margin-bottom: 16px; }}
 table {{ border-collapse: collapse; width: 100%; }}
 th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
 th {{ background: #f4f4f4; }}
 small {{ color: #555; }}
</style>
</head>
<body>

<h1>Retail Sales Forecast — Store {STORE_ID}</h1>
<div class="card">
  <p><b>Train range:</b> {train.index.min().date()} → {train.index.max().date()}<br/>
     <b>Test range:</b> {test.index.min().date()} → {test.index.max().date()}<br/>
     <b>Generated on:</b> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
  </p>
</div>

<h2>Model Performance</h2>
<div class="card">
  {metrics_html}
  <p><small>MAPE = Mean Absolute Percentage Error, RMSE = Root Mean Squared Error.</small></p>
</div>

<h2>History</h2>
<div class="card">
  <img alt="history" src="data:image/png;base64,{img_history_b64}" style="max-width:100%; height:auto;"/>
</div>

{img_compare_block}

<h2>Notes & Recommendations</h2>
<div class="card">
  <ul>
    <li>If <code>SARIMAX</code> yields lower MAPE/RMSE, its weekly seasonality fit is stronger on this store.</li>
    <li>If <code>Prophet</code> is better, try adding regressors (Holiday_Flag, Temperature, CPI) for more realism.</li>
    <li>For production: retrain on full history and forecast next N weeks; monitor errors weekly.</li>
  </ul>
</div>

</body>
</html>
"""

REPORT_NAME.write_text(report_html, encoding="utf-8")
print(f"\nReport saved → {REPORT_NAME.resolve()}")
print("Done.")
