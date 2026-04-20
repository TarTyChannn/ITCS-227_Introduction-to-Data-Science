"""
=======================================================================
  PC COMPONENTS — 10-YEAR FORECASTING
  Features: ARIMA(p,d,q) + Damped Holt's (optimized) + AR(3)+Seasonal
            Adaptive Inverse-Error Ensembling, Dual CI Bands (80%/95%),
            Heatmaps, Historical Event Annotations, Spectral Analysis.
  Outputs:  Line Charts, CSV Forecasts, Heatmaps, RMSE Audit Logs.
=======================================================================
"""

import os
import glob
import warnings
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. Centralized Configuration
# ---------------------------------------------------------
@dataclass
class PipelineConfig:
    force_rerun: bool = True

    # ARIMA Hyperparameters
    arima_p_candidates: list = field(default_factory=lambda: [2, 4, 6])
    arima_d_candidates: list = field(default_factory=lambda: [0, 1])
    arima_q_candidates: list = field(default_factory=lambda: [0, 2])
    arima_ridge_alpha:  float = 5.0

    # Damped Holt's Hyperparameters
    holts_phi:          float = 0.88 
    holts_alpha_range:  tuple = (0.1, 0.91, 0.10)
    holts_beta_range:   tuple = (0.05, 0.41, 0.05)

    # AR + Seasonal Hyperparameters
    ar_seasonal_lags:   int   = 3
    ar_seasonal_alpha:  float = 15.0

    # Pipeline Settings
    forecast_months:    int   = 120
    validation_months:  int   = 12
    min_history_months: int   = 19    # validation_months + ar_lags + safety buffer

    # Macro-Economic Decay Rates (Moore's Law)
    gpu_decay:          float = 0.01  # 1%  — prices stay rigid or inflate slightly
    ram_decay:          float = 0.08  # 8%  — commodity, tech advances quickly
    ssd_decay:          float = 0.10  # 10% — NAND gets cheaper steadily

    output_dirs: dict = field(default_factory=lambda: {
        'GPU':  'output/GPU',
        'RAM':  'output/RAM',
        'SSD':  'output/SSD',
        'LOGS': 'output/Logs'
    })

CONFIG = PipelineConfig()

for path in CONFIG.output_dirs.values():
    os.makedirs(path, exist_ok=True)

# ---------------------------------------------------------
# 1.5 Historical Market Shocks (For Annotations)
# ---------------------------------------------------------
HISTORICAL_EVENTS = {
    'GPU': [
        {'date': '2021-05-15', 'label': 'Crypto Boom Peak'},
        {'date': '2022-09-15', 'label': 'ETH Merge (Mining Ends)'},
        {'date': '2023-05-01', 'label': 'AI Demand Surge'}
    ],
    'RAM': [
        {'date': '2023-09-01', 'label': 'Samsung/Micron Production Cuts'}
    ],
    'SSD': [
        {'date': '2023-09-01', 'label': 'Global NAND Production Cuts'}
    ]
}

# ---------------------------------------------------------
# 2. Data Preprocessors
# ---------------------------------------------------------
def load_latest_csv(prefixes):
    """Finds and loads the most recently modified CSV from explicit prefixes.
    Stage 1 searches Datasets/ — falls back to root only if nothing found."""
    if isinstance(prefixes, str): prefixes = [prefixes]
    files = []

    # Stage 1: Exhaust all prefixes in Datasets/ directory first
    for prefix in prefixes:
        files.extend(glob.glob(f'Datasets/{prefix}*.csv'))

    # Stage 2: Fallback to root ONLY if Datasets/ returned nothing
    if not files:
        for prefix in prefixes:
            files.extend(glob.glob(f'{prefix}*.csv'))

    if not files:
        print(f"  [!] No CSV found for {prefixes}. Skipping...")
        return None

    latest_file = max(files, key=os.path.getmtime)
    df = pd.read_csv(latest_file)
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.tz_localize(None)
    df = df.dropna(subset=['date', 'avg_price'])
    df = df[df['avg_price'] > 0]
    return df

def preprocess_gpu():
    df = load_latest_csv(['gpu_price', 'gpu_prices'])
    if df is None: return None

    def get_brand(cat):
        c = str(cat).upper()
        if 'RTX' in c or 'GTX' in c: return 'NVIDIA'
        if 'RX' in c or 'RADEON' in c: return 'AMD'
        return 'Other'

    def get_series(cat):
        c = str(cat).upper()
        if 'RTX 30' in c: return 'RTX 3000 Series'
        if 'RTX 40' in c: return 'RTX 4000 Series'
        if 'RTX 50' in c: return 'RTX 5000 Series'
        if 'RX 6' in c:   return 'RX 6000 Series'
        if 'RX 7' in c:   return 'RX 7000 Series'
        if 'RX 9' in c:   return 'RX 9000 Series'
        return 'Other'

    df['Brand']  = df['category'].apply(get_brand)
    df['Series'] = df['category'].apply(get_series)
    return df

def preprocess_ram():
    df = load_latest_csv(['ram_price', 'memory_price', 'ram_prices', 'memory_prices'])
    if df is None: return None
    df['Size']       = df['category'].apply(lambda x: str(x).split()[0] if 'GB' in str(x) else 'Unknown')
    df['Generation'] = df['category'].apply(lambda x: str(x).split()[1] if len(str(x).split()) > 1 else 'Unknown')
    return df

def preprocess_ssd():
    df = load_latest_csv(['ssd_price', 'ssd_prices'])
    if df is None: return None

    def get_capacity(cat):
        cat = str(cat).upper()
        for cap in ['120GB', '240GB', '250GB', '256GB', '480GB', '500GB', '512GB', '1TB', '2TB', '4TB']:
            if cap in cat: return cap
        return 'Other'

    def get_interface(cat):
        cat = str(cat).upper()
        if 'NVME' in cat or 'PCIE' in cat: return 'PCIe'
        if 'SATA' in cat: return 'SATA'
        return 'Other'

    df['Capacity']  = df['category'].apply(get_capacity)
    df['Interface'] = df['category'].apply(get_interface)
    return df

# ---------------------------------------------------------
# 3. Core Forecasting Engine (3-Model Ensemble)
# ---------------------------------------------------------

def _run_arima(data, steps_out, p=4, d=1, q=2):
    """ARIMA(p,d,q) implemented via Ridge regression on differenced series."""
    y      = np.array(data, dtype=float)
    y_diff = np.diff(y, n=d)

    X_ar, Y_ar = [], []
    for i in range(p, len(y_diff)):
        X_ar.append(y_diff[i-p:i][::-1])
        Y_ar.append(y_diff[i])

    if len(X_ar) == 0:
        return np.full(steps_out, y[-1])

    X_ar, Y_ar = np.array(X_ar), np.array(Y_ar)
    ar_model   = Ridge(alpha=CONFIG.arima_ridge_alpha).fit(X_ar, Y_ar)
    ar_fitted  = ar_model.predict(X_ar)
    residuals  = Y_ar - ar_fitted

    # MA Component
    X_ma, Y_ma = [], []
    for i in range(q, len(residuals)):
        X_ma.append(residuals[i-q:i])
        Y_ma.append(residuals[i])
    ma_model = None
    if len(X_ma) > 0:
        ma_model = Ridge(alpha=CONFIG.arima_ridge_alpha).fit(np.array(X_ma), np.array(Y_ma))

    hist_diff = list(y_diff)
    hist_res  = list(residuals)
    preds_diff = []
    # +1e-5 prevents zero-variance crash on flat datasets
    max_delta = (np.std(y_diff) + 1e-5) * 3

    for _ in range(steps_out):
        x_ar = np.array(hist_diff[-p:][::-1]).reshape(1, -1)
        ar_p = ar_model.predict(x_ar)[0]
        ma_p = 0.0
        if ma_model is not None and len(hist_res) >= q:
            x_ma = np.array(hist_res[-q:]).reshape(1, -1)
            ma_p = ma_model.predict(x_ma)[0]
        pred = np.clip(ar_p + ma_p, -max_delta, max_delta)
        preds_diff.append(pred)
        hist_diff.append(pred)
        hist_res.append(0.0)

    result = list(y[-d:]) if d > 0 else [y[-1]]
    for delta in preds_diff:
        result.append(result[-1] + delta)
    preds = np.array(result[d if d > 0 else 1:])
    return np.clip(preds, y[-1] * 0.10, y[-1] * 15.0)


def _select_arima_order(series, val_months=12):
    """Grid-searches (p, d, q) combinations and returns best by validation RMSE."""
    if len(series) < 2 * val_months + 10:
        return (4, 1, 2)
    inner_train  = series[:-val_months]
    inner_actual = series[-val_months:]
    best_rmse, best_params = np.inf, (4, 1, 2)
    for p in CONFIG.arima_p_candidates:
        for d in CONFIG.arima_d_candidates:
            for q in CONFIG.arima_q_candidates:
                try:
                    fc_val = _run_arima(inner_train, val_months, p, d, q)
                    r = np.sqrt(mean_squared_error(inner_actual, fc_val))
                    if r < best_rmse:
                        best_rmse, best_params = r, (p, d, q)
                except Exception:
                    pass
    return best_params


def _run_holts(data, steps_out):
    """Damped Holt's Double Exponential Smoothing with optimized alpha/beta via grid search."""
    y   = np.array(data, dtype=float)
    phi = CONFIG.holts_phi

    def _sse(alpha, beta):
        if not (0 < alpha < 1 and 0 < beta < 1): return 1e10
        L = y[0]
        T = np.mean(np.diff(y[:4])) if len(y) >= 4 else \
            (y[-1] - y[0]) / max(len(y) - 1, 1)
        sse = 0.0
        for i in range(1, len(y)):
            L_p, T_p = L, T
            L    = alpha * y[i] + (1 - alpha) * (L_p + phi * T_p)
            T    = beta  * (L - L_p) + (1 - beta) * phi * T_p
            sse += (y[i] - (L_p + phi * T_p)) ** 2
        return sse

    # Grid search for optimal smoothing parameters
    best_sse, best_a, best_b = np.inf, 0.3, 0.1
    a_start, a_stop, a_step = CONFIG.holts_alpha_range
    b_start, b_stop, b_step = CONFIG.holts_beta_range
    for a in np.arange(a_start, a_stop, a_step):
        for b in np.arange(b_start, b_stop, b_step):
            s = _sse(a, b)
            if s < best_sse:
                best_sse, best_a, best_b = s, a, b

    # Re-run with best params to get final L, T state
    L = y[0]
    T = np.mean(np.diff(y[:4])) if len(y) >= 4 else \
        (y[-1] - y[0]) / max(len(y) - 1, 1)
    for i in range(1, len(y)):
        L_p, T_p = L, T
        L = best_a * y[i] + (1 - best_a) * (L_p + phi * T_p)
        T = best_b * (L - L_p) + (1 - best_b) * phi * T_p

    # Project forward using damped trend accumulation
    preds, phi_cumsum = [], 0.0
    for _ in range(steps_out):
        phi_cumsum = phi_cumsum * phi + phi
        preds.append(L + phi_cumsum * T)

    return np.clip(np.array(preds), y[-1] * 0.10, y[-1] * 15.0), best_a, best_b


def _run_ar_seasonal(data, steps_out, p=12):
    """AR(p) with Fourier seasonal features at 12, 24, 36, 60-month cycles."""
    y = np.array(data, dtype=float)
    n = len(y)
    if n <= p:
        return np.full(steps_out, y[-1])

    def _make_row(i, hist):
        lag_feats = hist[i-p:i][::-1]
        t_norm    = i / n
        sine12,  cos12  = np.sin(2 * np.pi * i / 12),  np.cos(2 * np.pi * i / 12)
        sine24,  cos24  = np.sin(2 * np.pi * i / 24),  np.cos(2 * np.pi * i / 24)
        sine36,  cos36  = np.sin(2 * np.pi * i / 36),  np.cos(2 * np.pi * i / 36)
        sine60,  cos60  = np.sin(2 * np.pi * i / 60),  np.cos(2 * np.pi * i / 60)
        return list(lag_feats) + [t_norm, sine12, cos12, sine24, cos24, sine36, cos36, sine60, cos60]

    X, Y = [], []
    for i in range(p, n):
        X.append(_make_row(i, y))
        Y.append(y[i])
    model = Ridge(alpha=CONFIG.ar_seasonal_alpha).fit(np.array(X), np.array(Y))

    history, preds = list(y), []
    for s in range(steps_out):
        row  = _make_row(n + s, np.array(history))
        pred = model.predict([row])[0]
        preds.append(pred)
        history.append(pred)

    return np.clip(np.array(preds), y[-1] * 0.10, y[-1] * 15.0)


def _adaptive_ensemble(fc_a, fc_h, fc_ar, r_a, r_h, r_ar):
    """Inverse-error weighted ensemble across 3 models."""
    eps = 1e-6
    inv = np.array([1 / (r_a + eps), 1 / (r_h + eps), 1 / (r_ar + eps)])
    w   = inv / inv.sum()
    return w[0] * fc_a + w[1] * fc_h + w[2] * fc_ar, w


def generate_forecast(series, steps, decay_rate):
    """
    Full 3-model ensemble forecast with:
    - ARIMA(p,d,q) with auto order selection
    - Damped Holt's with optimized alpha/beta
    - AR(12) + Fourier seasonal features
    - Adaptive inverse-error weighting
    - Moore's Law decay curve
    - Dual 80%/95% confidence intervals
    Returns: (forecast, ci_lo_80, ci_hi_80, ci_lo_95, ci_hi_95, final_rmse, weight_str)
    """
    y = np.array(series, dtype=float)

    if len(y) < CONFIG.min_history_months:
        flat = np.full(steps, y[-1])
        print(f"      [Warning] Insufficient data ({len(y)} pts). Returning flatline.")
        return flat, flat, flat, flat, flat, 0.0, "Flatline"

    train  = y[:-CONFIG.validation_months]
    actual = y[-CONFIG.validation_months:]

    # --- Auto-select ARIMA order on train slice ---
    best_p, best_d, best_q = _select_arima_order(train, CONFIG.validation_months)

    # --- Validation forecasts (with individual try/catch per model) ---
    try:
        val_arima = _run_arima(train, CONFIG.validation_months, best_p, best_d, best_q)
    except Exception as e:
        print(f"      [Warning] ARIMA validation failed: {e}")
        val_arima = np.full(CONFIG.validation_months, train[-1])

    try:
        val_holts, _, _ = _run_holts(train, CONFIG.validation_months)
    except Exception as e:
        print(f"      [Warning] Holt's validation failed: {e}")
        val_holts = np.full(CONFIG.validation_months, train[-1])

    try:
        val_ar = _run_ar_seasonal(train, CONFIG.validation_months, CONFIG.ar_seasonal_lags)
    except Exception as e:
        print(f"      [Warning] AR+Seasonal validation failed: {e}")
        val_ar = np.full(CONFIG.validation_months, train[-1])

    r_arima = np.sqrt(mean_squared_error(actual, val_arima))
    r_holts = np.sqrt(mean_squared_error(actual, val_holts))
    r_ar    = np.sqrt(mean_squared_error(actual, val_ar))

    val_ens, w_val = _adaptive_ensemble(val_arima, val_holts, val_ar, r_arima, r_holts, r_ar)
    final_rmse     = np.sqrt(mean_squared_error(actual, val_ens))

    # --- Full future forecasts on complete history ---
    try:
        fc_arima = _run_arima(y, steps, best_p, best_d, best_q)
    except Exception as e:
        print(f"      [Warning] ARIMA forecast failed: {e}")
        fc_arima = np.full(steps, y[-1])

    try:
        fc_holts, _, _ = _run_holts(y, steps)
    except Exception as e:
        print(f"      [Warning] Holt's forecast failed: {e}")
        fc_holts = np.full(steps, y[-1])

    try:
        fc_ar = _run_ar_seasonal(y, steps, CONFIG.ar_seasonal_lags)
    except Exception as e:
        print(f"      [Warning] AR+Seasonal forecast failed: {e}")
        fc_ar = np.full(steps, y[-1])

    # Apply Moore's Law decay before ensembling
    monthly_decay = (1 - decay_rate) ** (1 / 12)
    decay_curve   = np.power(monthly_decay, np.arange(1, steps + 1))
    fc_arima *= decay_curve
    fc_holts *= decay_curve
    fc_ar    *= decay_curve

    # Adaptive ensemble using validation-derived weights
    ensemble, w_full = _adaptive_ensemble(fc_arima, fc_holts, fc_ar, r_arima, r_holts, r_ar)
    forecast = np.clip(ensemble, y[-1] * 0.10, y[-1] * 5.0)

    # --- Dual Confidence Intervals (80% and 95%) ---
    val_resid_std   = np.std(actual - val_ens)
    horizon         = np.arange(1, steps + 1)
    ci_lo_80 = np.clip(forecast - val_resid_std * np.sqrt(horizon / 4) * 1.28, 0, None)
    ci_hi_80 =         forecast + val_resid_std * np.sqrt(horizon / 4) * 1.28
    ci_lo_95 = np.clip(forecast - val_resid_std * np.sqrt(horizon / 4) * 1.96, 0, None)
    ci_hi_95 =         forecast + val_resid_std * np.sqrt(horizon / 4) * 1.96

    weight_str = f"A:{w_full[0]:.2f}/H:{w_full[1]:.2f}/AR:{w_full[2]:.2f}"
    return forecast, ci_lo_80, ci_hi_80, ci_lo_95, ci_hi_95, final_rmse, weight_str


# ---------------------------------------------------------
# 3.5 Spectral Analysis (Dominant Price Cycles)
# ---------------------------------------------------------
def run_spectral_analysis(history_dict, component_name, output_dir):
    """Runs a periodogram on each category to identify dominant price cycles."""
    print(f"\n  [{component_name}] Dominant price cycles detected (periodogram):")
    cycle_logs = []
    for cat, series in history_dict.items():
        if len(series) < 24: continue
        y_centered = series - series.mean()
        freqs, power = periodogram(y_centered, fs=1.0)
        valid  = freqs > 0
        top3   = np.argsort(power[valid])[-3:][::-1]
        periods = 1 / freqs[valid][top3]
        cycle_str = ', '.join([f'~{p:.0f}mo' for p in periods])
        print(f"    {cat:20s}: {cycle_str}")
        cycle_logs.append({'Component': component_name, 'Category': cat, 'Top Cycles': cycle_str})

    if cycle_logs:
        cycle_path = os.path.join(output_dir, f'{component_name}_spectral_cycles.csv')
        pd.DataFrame(cycle_logs).to_csv(cycle_path, index=False)
        print(f"    [+] Spectral log saved to: {cycle_path}")


# ---------------------------------------------------------
# 3.6 Heatmap Generator
# ---------------------------------------------------------
def generate_change_heatmap(csv_data, base_prices, fc_index, title_group, output_filename):
    """Generates a Red/Green Heatmap showing % price change vs today, year-by-year."""
    years        = [fc_index[0].year + i for i in range(int(CONFIG.forecast_months / 12))]
    month_indices = [min(i * 12 + 11, len(fc_index) - 1) for i in range(len(years))]

    cats = [k.replace('_Forecast', '') for k in csv_data.keys() if '_Forecast' in k]
    if not cats: return

    # Pre-fill with NaN — missing rows render as grey "N/A", not fake 0%
    pct_matrix = np.full((len(cats), len(years)), np.nan)

    for i, cat in enumerate(cats):
        if cat not in base_prices: continue
        base = base_prices[cat]
        fc   = csv_data[f"{cat}_Forecast"]
        for j, m_idx in enumerate(month_indices):
            pct_matrix[i, j] = ((fc[m_idx] - base) / base) * 100

    plt.figure(figsize=(12, max(6, len(cats) * 0.4)), facecolor='#0e0e1c')
    ax = plt.gca()
    ax.set_facecolor('#0e0e1c')

    cmap = plt.cm.RdYlGn_r.copy()
    cmap.set_bad(color='#2a2a40')  # Dark neutral for missing data

    im = ax.imshow(pct_matrix, cmap=cmap, aspect='auto', vmin=-50, vmax=50)

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, color='white', fontsize=10)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, color='white', fontsize=10)

    for i in range(len(cats)):
        for j in range(len(years)):
            val = pct_matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "N/A", ha='center', va='center', color='#555588', fontsize=8, fontweight='bold')
            else:
                color = 'white' if abs(val) > 25 else 'black'
                ax.text(j, i, f"{val:+.0f}%", ha='center', va='center', color=color, fontsize=8, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='white')
    cbar.set_label('Price Change vs Today (%)', color='white')

    t_group = title_group if title_group != 'category' else 'All Models'
    plt.title(f"10-Year Price Change Heatmap ({t_group})", color='white', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=150, facecolor='#0e0e1c')
    plt.close()


# ---------------------------------------------------------
# 4. Batch Processing Factory
# ---------------------------------------------------------
def process_segment(df, component_name, group_column, decay_rate, output_dir):
    """Groups data, runs 3-model ensemble forecasts, plots results, exports CSV and heatmap."""
    if df is None: return []

    base_filename = f"{output_dir}/{component_name}_Forecast_By_{group_column}"

    if not CONFIG.force_rerun and os.path.exists(f"{base_filename}.png"):
        print(f"\n[{component_name}] Skipping {group_column} (Already exists)")
        return []

    print(f"\n[{component_name}] Running models grouped by {group_column}...")
    print(f"    {'Category':<20} | {'ARIMA':>7} | {'Holts':>7} | {'AR+Sea':>7} | {'Ens':>7} | Weights(A/H/AR)")
    print("    " + "-" * 75)

    monthly    = df.groupby([group_column, pd.Grouper(key='date', freq='MS')])['avg_price'].mean().reset_index()
    categories = monthly[group_column].unique()

    last_date = monthly['date'].max()
    fc_index  = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=CONFIG.forecast_months, freq='MS')

    plt.figure(figsize=(14, 7), facecolor='#0e0e1c')
    ax = plt.gca()
    ax.set_facecolor('#0e0e1c')
    ax.spines[['top', 'right', 'left', 'bottom']].set_color('#1a1a30')
    ax.tick_params(colors='#e8e8ff')
    ax.grid(True, color='#1a1a30', lw=0.5)

    colors     = plt.cm.tab20.colors
    csv_data   = {}
    audit_logs = []
    history_dict = {}  # For spectral analysis

    for i, cat in enumerate(categories):
        if cat in ['Other', 'Unknown']: continue

        hist = monthly[monthly[group_column] == cat].set_index('date')['avg_price'].sort_index().dropna()
        if len(hist) < 6: continue

        history_dict[cat] = hist.values

        fc, ci_lo_80, ci_hi_80, ci_lo_95, ci_hi_95, rmse_val, weights = \
            generate_forecast(hist.values, CONFIG.forecast_months, decay_rate)

        print(f"    -> {cat:<18} | RMSE: ${rmse_val:<6.2f} | Weights: {weights}")

        audit_logs.append({
            'Component': component_name,
            'Grouping':  group_column,
            'Category':  cat,
            'RMSE':      round(rmse_val, 2),
            'Weights':   weights
        })

        c = colors[i % len(colors)]
        ax.plot(hist.index, hist.values, color=c, lw=2, label=f"{cat}")
        ax.plot(fc_index, fc, color=c, lw=2, linestyle='--', alpha=0.9)

        # Dual CI bands: 95% outer (faint), 80% inner (visible)
        ax.fill_between(fc_index, ci_lo_95, ci_hi_95, color=c, alpha=0.08)
        ax.fill_between(fc_index, ci_lo_80, ci_hi_80, color=c, alpha=0.18)

        ax.text(fc_index[-1], fc[-1], f" ${fc[-1]:.0f}", color=c, fontsize=9, va='center')

        csv_data[f"{cat}_Forecast"]  = fc
        csv_data[f"{cat}_CI_Lo_80"]  = ci_lo_80
        csv_data[f"{cat}_CI_Hi_80"]  = ci_hi_80
        csv_data[f"{cat}_CI_Lo_95"]  = ci_lo_95
        csv_data[f"{cat}_CI_Hi_95"]  = ci_hi_95

    # Early exit guard — prevents blank charts from being saved
    if not csv_data:
        print(f"    [!] No valid data for {group_column}. Skipping plot output.")
        plt.close()
        return []

    title_group = group_column if group_column != 'category' else 'All Models'
    plt.title(f"{component_name} 10-Year Price Forecast grouped by {title_group}", color='white', fontsize=16, pad=15)
    plt.ylabel("Average Price (USD)", color='white')

    if group_column == 'category':
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), facecolor='#13132a', edgecolor='#1a1a30', labelcolor='white', fontsize='x-small', ncol=2)
    else:
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), facecolor='#13132a', edgecolor='#1a1a30', labelcolor='white')

    # Finalize ylim BEFORE tight_layout so layout captures correct dimensions
    ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.15)
    y_top = ax.get_ylim()[1]

    # Historical event annotations — rightward offset stays within axes boundary
    if component_name in HISTORICAL_EVENTS:
        for event in HISTORICAL_EVENTS[component_name]:
            event_date = pd.to_datetime(event['date'])
            if monthly['date'].min() <= event_date <= last_date:
                plt.axvline(event_date, color='#ff5555', linestyle='-', lw=1.5, alpha=0.6)
                plt.text(event_date + pd.DateOffset(days=20), y_top * 0.85,
                         f" {event['label']} ", color='#ff5555',
                         rotation=90, va='top', ha='left', fontsize=9, fontweight='bold', alpha=0.8)

    plt.axvline(last_date, color='#555588', linestyle=':', lw=2)
    plt.text(last_date, y_top * 0.95, " TODAY ", color='#555588', ha='right', va='top')

    plt.tight_layout()
    plt.savefig(f"{base_filename}.png", dpi=150, facecolor='#0e0e1c')
    plt.close()

    # Export forecast CSV
    if csv_data:
        df_out = pd.DataFrame(csv_data, index=fc_index)
        df_out.index.name = "Date"
        df_out.to_csv(f"{base_filename}.csv")

        # Build base prices for heatmap
        base_prices = {}
        for cat in categories:
            if cat in ['Other', 'Unknown']: continue
            hist = monthly[monthly[group_column] == cat].set_index('date')['avg_price'].dropna()
            if len(hist) >= 6: base_prices[cat] = hist.iloc[-1]

        generate_change_heatmap(csv_data, base_prices, fc_index, group_column, f"{base_filename}_Heatmap.png")
        print(f"  [+] Exported PNG, CSV, and Heatmap to {output_dir}")

    # Spectral analysis per segment
    if history_dict:
        run_spectral_analysis(history_dict, component_name, output_dir)

    return audit_logs


# ---------------------------------------------------------
# 5. Main Execution Block & Summary
# ---------------------------------------------------------
if __name__ == "__main__":
    print("=======================================================")
    print("  STARTING BATCH FORECAST PIPELINE (V5.1 - Conservative)")
    print("  Models: ARIMA(auto) · Damped Holt's · AR(3)+Seasonal [conservative]")
    print("=======================================================")

    master_audit_log = []

    # --- GPUs ---
    gpu_df = preprocess_gpu()
    master_audit_log.extend(process_segment(gpu_df, "GPU", "category", CONFIG.gpu_decay, CONFIG.output_dirs['GPU']))
    master_audit_log.extend(process_segment(gpu_df, "GPU", "Brand",    CONFIG.gpu_decay, CONFIG.output_dirs['GPU']))
    master_audit_log.extend(process_segment(gpu_df, "GPU", "Series",   CONFIG.gpu_decay, CONFIG.output_dirs['GPU']))

    # --- RAM ---
    ram_df = preprocess_ram()
    master_audit_log.extend(process_segment(ram_df, "RAM", "category",   CONFIG.ram_decay, CONFIG.output_dirs['RAM']))
    master_audit_log.extend(process_segment(ram_df, "RAM", "Generation", CONFIG.ram_decay, CONFIG.output_dirs['RAM']))
    master_audit_log.extend(process_segment(ram_df, "RAM", "Size",       CONFIG.ram_decay, CONFIG.output_dirs['RAM']))

    # --- SSDs ---
    ssd_df = preprocess_ssd()
    master_audit_log.extend(process_segment(ssd_df, "SSD", "category",  CONFIG.ssd_decay, CONFIG.output_dirs['SSD']))
    master_audit_log.extend(process_segment(ssd_df, "SSD", "Capacity",  CONFIG.ssd_decay, CONFIG.output_dirs['SSD']))
    master_audit_log.extend(process_segment(ssd_df, "SSD", "Interface", CONFIG.ssd_decay, CONFIG.output_dirs['SSD']))

    # Print structured summary table and save master audit log
    if master_audit_log:
        audit_df   = pd.DataFrame(master_audit_log)
        audit_path = os.path.join(CONFIG.output_dirs['LOGS'], 'rmse_audit_log.csv')
        audit_df.to_csv(audit_path, index=False)

        print("\n=======================================================")
        print("                 PIPELINE RUN SUMMARY")
        print("=======================================================")

        summary = audit_df.groupby(['Component', 'Grouping']).agg(
            Categories=('Category', 'count'),
            Avg_RMSE=('RMSE', 'mean')
        ).reset_index()

        print(f"{'Component':<12} | {'Grouping':<15} | {'Categories':<12} | {'Avg RMSE'}")
        print("-" * 60)
        for _, row in summary.iterrows():
            print(f"{row['Component']:<12} | {row['Grouping']:<15} | {row['Categories']:<12} | ${row['Avg_RMSE']:.2f}")

        print(f"\n[+] Detailed RMSE audit saved to: {audit_path}")

    print("=======================================================")
    print("  BATCH JOB COMPLETE. Check the /output directories.")
    print("=======================================================")
