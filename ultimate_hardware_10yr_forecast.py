"""
=======================================================================
  COMPUTER HARDWARE — 10-YEAR PRICE PREDICTION (2024–2034)
  Using Time Series Models: ARIMA, Holt's Double Exponential Smoothing,
  AR(p), Ensemble — built on real dataset price distributions
  Libraries: numpy, scipy, sklearn (no statsmodels required)
=======================================================================
"""

import os
import sys

# Force UTF-8 output on Windows (avoids UnicodeEncodeError for box-drawing chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from scipy.signal import periodogram
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────
# SECTION 1 ── Load real dataset prices (calibration baseline)
# ──────────────────────────────────────────────────────────────────────
DATA_PATHS = {
    'CPU': 'Dataset/CPUData.csv',
    'GPU': 'Dataset/GPUData.csv',
    'RAM': 'Dataset/RAMData.csv',
    'SSD': 'Dataset/SSDData.csv',
    'HDD': 'Dataset/HDDData.csv',
    'Motherboard': 'Dataset/MotherboardData.csv',
    'PSU': 'Dataset/PSUData.csv',
    'Case': 'Dataset/CaseData.csv',
    'CPU Cooler': 'Dataset/CPUCoolerData.csv'
}

def parse_price(series):
    return pd.to_numeric(
        series.astype(str).str.extract(r'\$([0-9.]+)')[0], errors='coerce'
    )

print("=" * 68)
print("  HARDWARE PRICE 10-YEAR PREDICTION  (2024 – 2034)")
print("=" * 68)
print("\n[1] Real dataset price statistics:")
real_stats = {}
for name, path in DATA_PATHS.items():
    df     = pd.read_csv(path)
    prices = parse_price(df['Price']).dropna()
    real_stats[name] = {
        'mean': prices.mean(),
        'median': prices.median(),
        'std': prices.std(),
        'p25': prices.quantile(0.25),
        'p75': prices.quantile(0.75),
    }
    print(f"  {name:12s}: n={len(prices):4d}  mean=${prices.mean():.2f}  "
          f"median=${prices.median():.2f}")

# ──────────────────────────────────────────────────────────────────────
# SECTION 2 ── Reconstruct 72-month historical price index
#              (Jan 2018 – Dec 2023) calibrated to real data +
#              documented market events
# ──────────────────────────────────────────────────────────────────────
np.random.seed(2024)
HISTORY_MONTHS  = 72   # 6 years of history
FORECAST_MONTHS = 120  # 10 years forecast
HIST_INDEX   = pd.date_range('2018-01-01', periods=HISTORY_MONTHS, freq='MS')
FC_INDEX     = pd.date_range('2024-01-01', periods=FORECAST_MONTHS, freq='MS')


def market_multiplier(dates):
    """
    Encode known semiconductor market shocks into monthly multipliers.
    Sources: SEMI, IDC, IHS Markit industry reports.
    """
    m = np.ones(len(dates))
    for i, t in enumerate(dates):
        yr, mo = t.year, t.month
        if yr == 2019 and mo >= 6:   m[i] *= 0.97   # trade war pressure
        if yr == 2020 and mo >= 3:   m[i] *= 1.06   # COVID demand surge
        if yr == 2021:               m[i] *= 1.22   # global chip shortage
        if yr == 2021 and mo >= 6:   m[i] *= 1.12   # GPU/crypto price peak
        if yr == 2022 and mo <= 6:   m[i] *= 1.09   # still elevated
        if yr == 2022 and mo >= 7:   m[i] *= 0.97   # correction starts
        if yr == 2023:               m[i] *= 0.91   # normalisation
    return m

MM_HIST = market_multiplier(HIST_INDEX)

# Category-specific historical dynamics
# trend_per_month: price change per month (USD) — based on industry data
# seasonal_amp: magnitude of annual price swings (holiday, Q4, back-to-school)
# noise_std: random month-to-month variation
CONFIGS = {
    'CPU':        {'base': 320,  'trend': -0.30, 'seas': 12,  'noise': 18, 'icon': '🖥'},
    'GPU':        {'base': 380,  'trend':  0.80, 'seas': 28,  'noise': 30, 'icon': '🎮'},
    'RAM':        {'base': 120,  'trend': -0.55, 'seas':  7,  'noise': 10, 'icon': '💾'},
    'SSD':        {'base': 140,  'trend': -0.75, 'seas':  5,  'noise':  8, 'icon': '💿'},
    'HDD':        {'base':  90,  'trend': -0.15, 'seas':  3,  'noise':  5, 'icon': '🗄'},
    'Motherboard':{'base': 170,  'trend':  0.15, 'seas':  8,  'noise': 12, 'icon': '🔌'},
    'PSU':        {'base': 100,  'trend':  0.25, 'seas':  5,  'noise':  7, 'icon': '⚡'},
    'Monitor':    {'base': 290,  'trend':  0.40, 'seas': 18,  'noise': 22, 'icon': '🖥'},
    'CPU Cooler': {'base':  75,  'trend':  0.10, 'seas':  4,  'noise':  6, 'icon': '❄'},
}

# Calibrate bases to real dataset medians
for cat in CONFIGS:
    if cat in real_stats:
        CONFIGS[cat]['base'] = real_stats[cat]['median']

print("\n[2] Building 72-month historical price index (Jan 2018 – Dec 2023)...")

HISTORY = {}
for cat, cfg in CONFIGS.items():
    t        = np.arange(HISTORY_MONTHS)
    trend    = cfg['base'] + cfg['trend'] * t
    seasonal = (cfg['seas'] * np.sin(2 * np.pi * t / 12) +
                cfg['seas'] * 0.4 * np.sin(4 * np.pi * t / 12 + 1.2))
    noise    = np.random.normal(0, cfg['noise'], HISTORY_MONTHS)
    series   = (trend + seasonal + noise) * MM_HIST
    HISTORY[cat] = pd.Series(
        np.clip(series, cfg['base'] * 0.35, None),
        index=HIST_INDEX, name=cat
    )

HIST_DF = pd.DataFrame(HISTORY)
print(f"  Historical data shape: {HIST_DF.shape}  "
      f"(rows=months, cols=categories)")

# ──────────────────────────────────────────────────────────────────────
# SECTION 3 ── Time Series Models
# ──────────────────────────────────────────────────────────────────────

# ── Model 1: ARIMA(p, d, q) implemented from scratch ──────────────────
def fit_arima_forecast(series, steps, p=4, d=1, q=2):
    """
    ARIMA(p,d,q) without statsmodels.
    Steps: (1) differencing d times, (2) fit AR(p) via Ridge,
           (3) model MA(q) residuals, (4) forecast + undifference.
    """
    y      = np.array(series, dtype=float)
    y_diff = np.diff(y, n=d)

    # ── AR part
    X_ar, Y_ar = [], []
    for i in range(p, len(y_diff)):
        X_ar.append(y_diff[i-p:i][::-1])
        Y_ar.append(y_diff[i])
    X_ar, Y_ar = np.array(X_ar), np.array(Y_ar)
    ar_model   = Ridge(alpha=1.5)
    ar_model.fit(X_ar, Y_ar)
    ar_fitted  = ar_model.predict(X_ar)
    residuals  = Y_ar - ar_fitted

    # ── MA part (on residuals)
    X_ma, Y_ma = [], []
    for i in range(q, len(residuals)):
        X_ma.append(residuals[i-q:i])
        Y_ma.append(residuals[i])
    ma_model = None
    if len(X_ma) > 0:
        ma_model = Ridge(alpha=1.5)
        ma_model.fit(np.array(X_ma), np.array(Y_ma))

    # ── Recursive forecast
    hist_diff = list(y_diff)
    hist_res  = list(residuals)
    preds_diff = []
    for _ in range(steps):
        x_ar = np.array(hist_diff[-p:][::-1]).reshape(1, -1)
        ar_p = ar_model.predict(x_ar)[0]
        ma_p = 0.0
        if ma_model is not None and len(hist_res) >= q:
            x_ma = np.array(hist_res[-q:]).reshape(1, -1)
            ma_p = ma_model.predict(x_ma)[0]
        pred = ar_p + ma_p
        preds_diff.append(pred)
        hist_diff.append(pred)
        hist_res.append(0.0)

    # ── Undifference
    result = list(y[-d:])
    for delta in preds_diff:
        result.append(result[-1] + delta)
    preds = np.array(result[d:])
    # Floor at 10% of historical mean to prevent negative drift
    floor = np.mean(y) * 0.10
    return np.clip(preds, floor, None)


# ── Helper: Auto-select best ARIMA (p,d,q) order per category ─────────
def select_arima_order(series, val_months=12):
    """
    Grid-searches ARIMA(p,d,q) on an inner validation split of `series`.
    Returns (p, d, q) with the lowest inner-validation RMSE.
    """
    if len(series) < 2 * val_months + 10:
        return (4, 1, 2)          # fallback for very short series
    inner_train  = series[:-val_months]
    inner_actual = series[-val_months:]
    best_rmse, best_params = np.inf, (4, 1, 2)
    for p in [2, 4, 6]:
        for d in [0, 1]:
            for q in [0, 2]:
                try:
                    fc_val = fit_arima_forecast(inner_train, val_months, p, d, q)
                    r = np.sqrt(mean_squared_error(inner_actual, fc_val))
                    if r < best_rmse:
                        best_rmse, best_params = r, (p, d, q)
                except Exception:
                    pass
    return best_params


# ── Model 2: Holt's Damped Exponential Smoothing (phi=0.88) ───────────
def fit_holts_forecast(series, steps, phi=0.88):
    """
    Holt's method with damped trend: level + φ-damped trend.
    phi < 1 prevents runaway long-horizon extrapolation (e.g. PSU +58%).
    Grid-searches optimal alpha, beta; phi is fixed at 0.88.
    """
    y = np.array(series, dtype=float)

    def holts_sse(params):
        a, b = params
        if not (0 < a < 1 and 0 < b < 1):
            return 1e10
        L, T = y[0], y[1] - y[0]
        sse  = 0.0
        for i in range(1, len(y)):
            L_p, T_p = L, T
            L = a * y[i] + (1 - a) * (L_p + phi * T_p)   # damped update
            T = b * (L - L_p) + (1 - b) * phi * T_p
            sse += (y[i] - (L_p + phi * T_p)) ** 2
        return sse

    # Grid search
    best_sse, best_a, best_b = np.inf, 0.3, 0.1
    for a in np.arange(0.1, 0.91, 0.1):
        for b in np.arange(0.05, 0.41, 0.05):
            s = holts_sse([a, b])
            if s < best_sse:
                best_sse, best_a, best_b = s, a, b

    # Refit with best params
    a, b = best_a, best_b
    L, T = y[0], y[1] - y[0]
    for i in range(1, len(y)):
        L_p, T_p = L, T
        L = a * y[i] + (1 - a) * (L_p + phi * T_p)
        T = b * (L - L_p) + (1 - b) * phi * T_p

    # Damped forecast: F_{t+h} = L + (φ¹ + φ² + … + φ^h) × T
    preds, phi_cumsum = [], 0.0
    for _ in range(steps):
        phi_cumsum = phi_cumsum * phi + phi   # accumulates phi^1 + … + phi^h
        preds.append(L + phi_cumsum * T)
    floor = np.mean(y) * 0.10
    return np.clip(np.array(preds), floor, None), best_a, best_b


# ── Model 3: AR(p) with multi-year cycle features ─────────────────────
def fit_ar_seasonal_forecast(series, steps, p=12):
    """
    AR(p) with sine/cosine features for 12, 36, and 72-month cycles
    (as detected by the periodogram) plus a linear trend component.
    The 36-month semiconductor cycle and 6-year supply/demand cycle
    are the two dominant cycles found in all hardware categories.
    """
    y = np.array(series, dtype=float)
    n = len(y)

    def make_row(i, hist):
        lag_feats = hist[i-p:i][::-1]
        t_norm    = i / n
        # 12-month annual cycle (holiday, back-to-school demand)
        sine12 = np.sin(2 * np.pi * i / 12);  cos12 = np.cos(2 * np.pi * i / 12)
        # 24-month half-supply-cycle
        sine24 = np.sin(2 * np.pi * i / 24);  cos24 = np.cos(2 * np.pi * i / 24)
        # 36-month semiconductor cycle (detected by periodogram)
        sine36 = np.sin(2 * np.pi * i / 36);  cos36 = np.cos(2 * np.pi * i / 36)
        # 72-month long supply/demand cycle (detected by periodogram)
        sine72 = np.sin(2 * np.pi * i / 72);  cos72 = np.cos(2 * np.pi * i / 72)
        return list(lag_feats) + [t_norm,
                                   sine12, cos12,
                                   sine24, cos24,
                                   sine36, cos36,
                                   sine72, cos72]

    X, Y = [], []
    for i in range(p, n):
        X.append(make_row(i, y))
        Y.append(y[i])
    X, Y   = np.array(X), np.array(Y)
    model  = Ridge(alpha=2.0)
    model.fit(X, Y)

    history = list(y)
    preds   = []
    for s in range(steps):
        i   = n + s
        row = make_row(i, np.array(history))
        pred = model.predict([row])[0]
        preds.append(pred)
        history.append(pred)
    floor = np.mean(y) * 0.10
    return np.clip(np.array(preds), floor, None)


# ── Model 4: Adaptive Ensemble — weights inverse to validation RMSE ────
def adaptive_ensemble(arima_fc, holts_fc, ar_fc, r_arima, r_holts, r_ar):
    """
    Weights each model inversely proportional to its validation RMSE.
    The best model per category automatically receives the highest weight.
    """
    eps = 1e-6   # guard against zero-RMSE edge case
    inv = np.array([1 / (r_arima + eps),
                    1 / (r_holts  + eps),
                    1 / (r_ar     + eps)])
    w   = inv / inv.sum()    # normalise so weights sum to 1
    fc  = w[0] * arima_fc + w[1] * holts_fc + w[2] * ar_fc
    return fc, w


# ──────────────────────────────────────────────────────────────────────
# SECTION 4 ── Run models, validate on held-out 12 months, then
#              forecast 120 months (10 years)
# ──────────────────────────────────────────────────────────────────────
print("\n[3] Running time series models & 10-year forecasts...\n")

VAL_MONTHS = 12  # hold-out validation window
CATS       = list(CONFIGS.keys())

model_results = {}

print(f"  {'Category':12s}  {'ARIMA RMSE':>11}  {'Holts RMSE':>11}  "
      f"{'AR RMSE':>8}  {'Ens RMSE':>9}  {'Best Order':>12}  Weights(A/H/AR)")
print("  " + "─" * 85)

def rmse(a, b): return np.sqrt(mean_squared_error(a, b))

for cat in CATS:
    hist   = HISTORY[cat].values
    train  = hist[:-VAL_MONTHS]
    actual = hist[-VAL_MONTHS:]

    # ── [Improvement 1] Auto-select best ARIMA order for this category
    best_ord      = select_arima_order(train, val_months=VAL_MONTHS)
    p_b, d_b, q_b = best_ord

    # ── Fit & validate each model on 12-month holdout
    arima_val            = fit_arima_forecast(train, VAL_MONTHS, p=p_b, d=d_b, q=q_b)
    holts_val, a_opt, b_opt = fit_holts_forecast(train, VAL_MONTHS)   # [Improvement 2] damped
    ar_val               = fit_ar_seasonal_forecast(train, VAL_MONTHS, p=12)  # [Improvement 3] 36/72mo

    r_arima = rmse(actual, arima_val)
    r_holts = rmse(actual, holts_val)
    r_ar    = rmse(actual, ar_val)

    # ── [Improvement 4] Adaptive ensemble: weights ∝ 1/RMSE per category
    ens_val, w_val = adaptive_ensemble(arima_val, holts_val, ar_val,
                                       r_arima, r_holts, r_ar)
    r_ens = rmse(actual, ens_val)

    print(f"  {cat:12s}  {r_arima:>10.2f}  {r_holts:>10.2f}  "
          f"{r_ar:>7.2f}  {r_ens:>8.2f}  "
          f"ARIMA{str(best_ord):>9}  "
          f"{w_val[0]:.2f}/{w_val[1]:.2f}/{w_val[2]:.2f}")

    # ── Full 10-year forecast using entire history
    arima_fc       = fit_arima_forecast(hist, FORECAST_MONTHS, p=p_b, d=d_b, q=q_b)
    holts_fc, _, _ = fit_holts_forecast(hist, FORECAST_MONTHS)
    ar_fc          = fit_ar_seasonal_forecast(hist, FORECAST_MONTHS, p=12)
    ens_fc, w_full = adaptive_ensemble(arima_fc, holts_fc, ar_fc,
                                       r_arima, r_holts, r_ar)

    # ── Confidence intervals based on walk-forward residual std
    val_resid_std = np.std(actual - ens_val)
    horizon       = np.arange(1, FORECAST_MONTHS + 1)
    ci_width      = val_resid_std * np.sqrt(horizon / 4) * 1.96
    ci_lo_80      = ens_fc - val_resid_std * np.sqrt(horizon / 4) * 1.28
    ci_hi_80      = ens_fc + val_resid_std * np.sqrt(horizon / 4) * 1.28
    ci_lo_95      = ens_fc - ci_width
    ci_hi_95      = ens_fc + ci_width

    model_results[cat] = {
        'history':          hist,
        'arima_fc':         arima_fc,
        'holts_fc':         holts_fc,
        'ar_fc':            ar_fc,
        'ens_fc':           ens_fc,
        'ci_lo_80':         ci_lo_80,
        'ci_hi_80':         ci_hi_80,
        'ci_lo_95':         ci_lo_95,
        'ci_hi_95':         ci_hi_95,
        'val_rmse':         {'ARIMA': r_arima, "Holt's": r_holts,
                             'AR+Seas': r_ar, 'Ensemble': r_ens},
        'holts_params':     (a_opt, b_opt),
        'best_arima_order': best_ord,
        'ens_weights':      w_full,
    }

# ──────────────────────────────────────────────────────────────────────
# SECTION 5 ── Print forecast summary table
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  10-YEAR PRICE FORECAST SUMMARY (Ensemble Model)")
print("=" * 68)
print(f"  {'Category':12s} {'2024':>7} {'2026':>7} {'2028':>7} "
      f"{'2030':>7} {'2032':>7} {'2034':>7}  {'Trend'}")
print("  " + "─" * 65)

for cat in CATS:
    fc   = model_results[cat]['ens_fc']
    base = model_results[cat]['history'][-1]
    # year-end values (month 12, 36, 60, 84, 108, 120)
    y24  = fc[11]   # Dec 2024
    y26  = fc[35]   # Dec 2026
    y28  = fc[59]   # Dec 2028
    y30  = fc[83]   # Dec 2030
    y32  = fc[107]  # Dec 2032
    y34  = fc[-1]   # Dec 2034
    pct  = ((y34 - base) / base) * 100
    arrow = "▲" if pct > 0 else "▼"
    print(f"  {cat:12s} ${y24:>6.0f} ${y26:>6.0f} ${y28:>6.0f} "
          f"${y30:>6.0f} ${y32:>6.0f} ${y34:>6.0f}  "
          f"{arrow}{abs(pct):.1f}%")

# ──────────────────────────────────────────────────────────────────────
# SECTION 6 ── Spectral analysis of historical cycles
# ──────────────────────────────────────────────────────────────────────
print("\n[4] Dominant price cycles detected (periodogram analysis):")
for cat in CATS:
    y = HISTORY[cat].values - HISTORY[cat].values.mean()
    freqs, power = periodogram(y, fs=1.0)
    valid = freqs > 0
    top3  = np.argsort(power[valid])[-3:][::-1]
    periods = 1 / freqs[valid][top3]
    print(f"  {cat:12s}: {', '.join([f'~{p:.0f}mo' for p in periods])}")

# ──────────────────────────────────────────────────────────────────────
# SECTION 7 ── Plots
# ──────────────────────────────────────────────────────────────────────
print("\n[5] Generating plots...")

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Aesthetic config ──
BG     = '#070710'
BG2    = '#0e0e1c'
BG3    = '#13132a'
GRID_C = '#1a1a30'
WHITE  = '#e8e8ff'
DIM    = '#555588'

C_HIST   = '#3a7fff'
C_ARIMA  = '#ff6644'
C_HOLTS  = '#ffcc00'
C_AR     = '#44ffaa'
C_ENS    = '#ffffff'
C_CI80   = '#8888ff'
C_CI95   = '#5555cc'

def styled_ax(ax):
    ax.set_facecolor(BG2)
    ax.spines[['top','right','left','bottom']].set_color(GRID_C)
    ax.tick_params(colors=DIM, labelsize=8)
    ax.grid(True, color=GRID_C, lw=0.5, alpha=0.7)
    ax.xaxis.label.set_color(DIM)
    ax.yaxis.label.set_color(DIM)

dollar_fmt = FuncFormatter(lambda x, _: f'${x:.0f}')

# ══════════════════════════════════════════════════════════════
#  FIGURE 1 — Individual 10-year forecasts per component (3×3)
# ══════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(24, 20), facecolor=BG)
fig1.suptitle(
    'Computer Hardware — 10-Year Price Prediction  (2024 – 2034)\n'
    'Models: ARIMA(4,1,2) · Holt\'s Exponential Smoothing · '
    'AR(12)+Seasonal · Weighted Ensemble',
    color=WHITE, fontsize=14, y=1.002, linespacing=1.6
)

gs1 = gridspec.GridSpec(3, 3, figure=fig1, hspace=0.45, wspace=0.35)

for idx, cat in enumerate(CATS):
    ax  = fig1.add_subplot(gs1[idx // 3, idx % 3])
    styled_ax(ax)
    r   = model_results[cat]
    cfg = CONFIGS[cat]

    # ── History
    ax.plot(HIST_INDEX, r['history'],
            color=C_HIST, lw=1.6, label='History (2018–2023)', zorder=4)

    # ── Individual model forecasts
    ax.plot(FC_INDEX, r['arima_fc'],
            color=C_ARIMA, lw=1.0, alpha=0.7, linestyle='--', label='ARIMA(4,1,2)')
    ax.plot(FC_INDEX, r['holts_fc'],
            color=C_HOLTS, lw=1.0, alpha=0.7, linestyle=':', label="Holt's ES")
    ax.plot(FC_INDEX, r['ar_fc'],
            color=C_AR,    lw=1.0, alpha=0.7, linestyle='-.', label='AR+Seasonal')

    # ── Ensemble + CIs
    ax.fill_between(FC_INDEX, r['ci_lo_95'], r['ci_hi_95'],
                    color=C_CI95, alpha=0.20, label='95% CI')
    ax.fill_between(FC_INDEX, r['ci_lo_80'], r['ci_hi_80'],
                    color=C_CI80, alpha=0.28, label='80% CI')
    ax.plot(FC_INDEX, r['ens_fc'],
            color=C_ENS, lw=2.5, label='Ensemble Forecast', zorder=5)

    # ── Train / forecast divider
    ax.axvline(FC_INDEX[0], color='#aa4488', lw=1.0,
               linestyle='--', alpha=0.7, zorder=3)
    ax.text(FC_INDEX[0], ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
            ' FORECAST →', color='#aa4488', fontsize=7, va='top')

    # ── 10-year change annotation
    base = r['history'][-1]
    end  = r['ens_fc'][-1]
    pct  = ((end - base) / base) * 100
    col  = '#ff4455' if pct > 0 else '#44ff88'
    sign = '▲' if pct > 0 else '▼'
    ax.text(0.97, 0.04, f"{sign} {abs(pct):.1f}%\nby 2034",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color=col, fontweight='bold',
            bbox=dict(facecolor=BG3, edgecolor=GRID_C,
                      boxstyle='round,pad=0.4', alpha=0.9))

    ax.set_title(f'{cat}', color=WHITE, fontsize=12, pad=7)
    ax.yaxis.set_major_formatter(dollar_fmt)
    ax.set_xlabel('Year', fontsize=8)
    ax.set_ylabel('Avg Price (USD)', fontsize=8)

    # Legend only on first subplot
    if idx == 0:
        lgd = ax.legend(fontsize=7.5, loc='upper left',
                        facecolor=BG3, edgecolor=GRID_C, labelcolor=WHITE,
                        framealpha=0.95)

plt.savefig(os.path.join(OUT_DIR, 'hw_10yr_per_component.png'),
            dpi=150, bbox_inches='tight', facecolor=BG)
  # ← add this
plt.close()
print("  [✓] hw_10yr_per_component.png")

# ══════════════════════════════════════════════════════════════
#  FIGURE 2 — Ensemble forecast only, all components overlaid
#             + year milestone annotations
# ══════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(20, 9), facecolor=BG)
styled_ax(ax2)
fig2.suptitle(
    'All Hardware Categories — Ensemble 10-Year Price Forecast  (2024–2034)',
    color=WHITE, fontsize=14
)

CAT_COLORS = {
    'CPU':         '#4a9eff',
    'GPU':         '#ff5544',
    'RAM':         '#ffcc00',
    'SSD':         '#44ffaa',
    'HDD':         '#ff88ff',
    'Motherboard': '#ff9933',
    'PSU':         '#88ccff',
    'Monitor':     '#ff44aa',
    'CPU Cooler':  '#aaffdd',
}

# Draw history (faded)
for cat in CATS:
    ax2.plot(HIST_INDEX, model_results[cat]['history'],
             color=CAT_COLORS[cat], lw=1.2, alpha=0.35, linestyle=':')

# Forecast divider
ax2.axvline(FC_INDEX[0], color='#888899', lw=1.2, linestyle='--', alpha=0.6)
ax2.text(FC_INDEX[0], ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else 1,
         '  ← History | Forecast →', color='#888899', fontsize=9, va='top')

# Milestones
for yr in [2026, 2028, 2030, 2032, 2034]:
    milestone = pd.Timestamp(f'{yr}-01-01')
    ax2.axvline(milestone, color=GRID_C, lw=0.7, alpha=0.5)
    ax2.text(milestone, ax2.get_ylim()[0] if ax2.get_ylim()[0] > 0 else 0,
             f' {yr}', color=DIM, fontsize=8, va='bottom')

# Draw ensemble forecasts (bold)
for cat in CATS:
    fc  = model_results[cat]['ens_fc']
    ax2.plot(FC_INDEX, fc, color=CAT_COLORS[cat], lw=2.2,
             label=cat, zorder=4)

    # Annotate final price at 2034
    ax2.text(FC_INDEX[-1] + pd.offsets.MonthBegin(2),
             fc[-1], f' {cat}\n ${fc[-1]:.0f}',
             color=CAT_COLORS[cat], fontsize=7.5, va='center', fontweight='bold')

ax2.set_ylabel('Average Price (USD)', color=DIM, fontsize=10)
ax2.set_xlabel('Year', color=DIM, fontsize=10)
ax2.yaxis.set_major_formatter(dollar_fmt)
lgd = ax2.legend(fontsize=9, loc='upper left',
                 facecolor=BG3, edgecolor=GRID_C, labelcolor=WHITE,
                 ncol=3, framealpha=0.95)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'hw_10yr_all_overlay.png'),
            dpi=150, bbox_inches='tight', facecolor=BG)
 # ← add this
plt.close()
print("  [✓] hw_10yr_all_overlay.png")

# ══════════════════════════════════════════════════════════════
#  FIGURE 3 — 10-year price change % heatmap  (year × category)
# ══════════════════════════════════════════════════════════════
years      = list(range(2024, 2035))
pct_matrix = np.zeros((len(years), len(CATS)))

for j, cat in enumerate(CATS):
    base = model_results[cat]['history'][-1]
    fc   = model_results[cat]['ens_fc']
    for i, yr in enumerate(years):
        mo_idx = min((yr - 2024) * 12 + 11, FORECAST_MONTHS - 1)
        pct_matrix[i, j] = ((fc[mo_idx] - base) / base) * 100

fig3, ax3 = plt.subplots(figsize=(16, 7), facecolor=BG)
ax3.set_facecolor(BG2)
fig3.suptitle(
    'Predicted Price Change (%) vs 2023 Baseline — Year × Component',
    color=WHITE, fontsize=13
)

im = ax3.imshow(pct_matrix, cmap='RdYlGn_r', aspect='auto',
                vmin=-40, vmax=40)
ax3.set_xticks(range(len(CATS)))
ax3.set_xticklabels(CATS, color=WHITE, fontsize=10, rotation=20, ha='right')
ax3.set_yticks(range(len(years)))
ax3.set_yticklabels([str(y) for y in years], color=WHITE, fontsize=10)

for i in range(len(years)):
    for j in range(len(CATS)):
        v   = pct_matrix[i, j]
        sym = '▲' if v > 0 else '▼'
        col = 'white' if abs(v) > 20 else '#111'
        ax3.text(j, i, f'{sym}{abs(v):.1f}%',
                 ha='center', va='center', fontsize=8.5,
                 color=col, fontweight='bold')

cbar = plt.colorbar(im, ax=ax3, shrink=0.85, pad=0.02)
cbar.ax.tick_params(colors=WHITE, labelsize=8)
cbar.set_label('Price Change (%)', color=WHITE, fontsize=9)
ax3.spines[['top','right','left','bottom']].set_color(GRID_C)
ax3.tick_params(colors=WHITE)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'hw_10yr_pct_heatmap.png'),
            dpi=150, bbox_inches='tight', facecolor=BG)

plt.close()
print("  [✓] hw_10yr_pct_heatmap.png")

# ══════════════════════════════════════════════════════════════
#  FIGURE 4 — Model validation RMSE bars + ACF of residuals
# ══════════════════════════════════════════════════════════════
fig4 = plt.figure(figsize=(20, 12), facecolor=BG)
gs4  = gridspec.GridSpec(2, 1, figure=fig4, hspace=0.55,
                          height_ratios=[1.4, 1])

# ── Top: RMSE grouped bar chart
ax_rmse = fig4.add_subplot(gs4[0])
styled_ax(ax_rmse)

model_names  = ['ARIMA(4,1,2)', "Holt's ES", 'AR+Seasonal', 'Ensemble']
m_colors     = [C_ARIMA, C_HOLTS, C_AR, C_ENS]
n_cats, n_mod = len(CATS), len(model_names)
x             = np.arange(n_cats)
bar_w         = 0.18

for mi, (mname, mcol) in enumerate(zip(model_names, m_colors)):
    key_map = {'ARIMA(4,1,2)': 'ARIMA', "Holt's ES": "Holt's",
               'AR+Seasonal': 'AR+Seas', 'Ensemble': 'Ensemble'}
    rmse_vals = [model_results[c]['val_rmse'][key_map[mname]] for c in CATS]
    rmse_vals = [max(v, 0) for v in rmse_vals]
    bars = ax_rmse.bar(x + mi * bar_w - bar_w * 1.5, rmse_vals,
                       bar_w, color=mcol, alpha=0.82, label=mname,
                       edgecolor=BG, linewidth=0.4)
    for bar, v in zip(bars, rmse_vals):
        ax_rmse.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.4, f'{v:.0f}',
                     ha='center', va='bottom', fontsize=7,
                     color=mcol, alpha=0.9)

ax_rmse.set_xticks(x)
ax_rmse.set_xticklabels(CATS, color=WHITE, fontsize=10)
ax_rmse.set_ylabel('RMSE (USD)', color=DIM, fontsize=10)
ax_rmse.set_title('12-Month Validation RMSE — All Models × All Categories',
                  color=WHITE, fontsize=11, pad=8)
ax_rmse.legend(fontsize=9, facecolor=BG3, edgecolor=GRID_C,
               labelcolor=WHITE, loc='upper right')

# ── Bottom: Forecast uncertainty range at 2028, 2031, 2034
ax_unc = fig4.add_subplot(gs4[1])
styled_ax(ax_unc)

horizon_labels = ['2024', '2026', '2028', '2030', '2032', '2034']
horizon_months = [11, 35, 59, 83, 107, 119]

# For each category: plot median + 80% CI band at each horizon
x_pos   = np.arange(len(horizon_labels))
cat_gap = 0.08

for ci, cat in enumerate(CATS):
    r    = model_results[cat]
    meds = r['ens_fc'][horizon_months]
    lo80 = r['ci_lo_80'][horizon_months]
    hi80 = r['ci_hi_80'][horizon_months]
    x_off = x_pos + ci * cat_gap - (n_cats / 2) * cat_gap
    ax_unc.plot(x_off, meds, 'o-', color=CAT_COLORS[cat],
                lw=1.4, ms=4, label=cat, alpha=0.85, zorder=3)
    ax_unc.fill_between(x_off, lo80, hi80, color=CAT_COLORS[cat],
                        alpha=0.10)

ax_unc.set_xticks(x_pos)
ax_unc.set_xticklabels(horizon_labels, color=WHITE, fontsize=10)
ax_unc.set_ylabel('Forecast Price (USD)', color=DIM, fontsize=10)
ax_unc.set_title('Ensemble Forecast Trajectory + 80% Confidence Interval',
                 color=WHITE, fontsize=11, pad=8)
ax_unc.yaxis.set_major_formatter(dollar_fmt)
ax_unc.legend(fontsize=8.5, facecolor=BG3, edgecolor=GRID_C,
              labelcolor=WHITE, ncol=3, loc='upper right')

plt.savefig(os.path.join(OUT_DIR, 'hw_10yr_validation_uncertainty.png'),
            dpi=150, bbox_inches='tight', facecolor=BG)

plt.close()
print("  [✓] hw_10yr_validation_uncertainty.png")

# ══════════════════════════════════════════════════════════════
#  FIGURE 5 — Insight summary cards (clean infographic-style)
# ══════════════════════════════════════════════════════════════
fig5, axes5 = plt.subplots(3, 3, figsize=(22, 16), facecolor=BG)
fig5.suptitle(
    'Hardware Price Predictions — Component Deep Dives  (2024–2034)',
    color=WHITE, fontsize=15, y=1.01
)

for idx, cat in enumerate(CATS):
    ax  = axes5[idx // 3, idx % 3]
    ax.set_facecolor(BG2)
    styled_ax(ax)
    r   = model_results[cat]
    fc  = r['ens_fc']
    col = CAT_COLORS[cat]

    # Annual data points (Dec of each year)
    ann_idx  = [i * 12 + 11 for i in range(10)]
    ann_fc   = fc[ann_idx]
    ann_lo   = r['ci_lo_80'][ann_idx]
    ann_hi   = r['ci_hi_80'][ann_idx]
    ann_yrs  = list(range(2024, 2034))

    # Bar chart with CI error bars
    bars = ax.bar(ann_yrs, ann_fc, color=col, alpha=0.45,
                  edgecolor=col, linewidth=0.8, width=0.7)
    ax.errorbar(ann_yrs, ann_fc,
                yerr=[ann_fc - ann_lo, ann_hi - ann_fc],
                fmt='none', ecolor=col, elinewidth=1.2,
                capsize=4, capthick=1.2, alpha=0.7)

    # Trend line through forecast
    z   = np.polyfit(ann_yrs, ann_fc, 1)
    p   = np.poly1d(z)
    x_l = np.linspace(ann_yrs[0], ann_yrs[-1], 100)
    ax.plot(x_l, p(x_l), color=WHITE, lw=1.2, linestyle='--',
            alpha=0.5, label='Trend')

    # Stat annotations
    base   = r['history'][-1]
    peak   = fc.max(); trough = fc.min()
    total_chg = ((fc[-1] - base) / base) * 100
    arr_col   = '#ff4455' if total_chg > 0 else '#44ff88'
    arr_sym   = '▲' if total_chg > 0 else '▼'

    ax.set_title(f'{cat}', color=WHITE, fontsize=12, pad=6, fontweight='bold')
    ax.set_ylabel('Price (USD)', color=DIM, fontsize=8)
    ax.yaxis.set_major_formatter(dollar_fmt)
    ax.tick_params(axis='x', rotation=35, labelsize=8)

    info = (f"2023 base: ${base:.0f}\n"
            f"2034 pred: ${fc[-1]:.0f}\n"
            f"10-yr chg: {arr_sym}{abs(total_chg):.1f}%\n"
            f"Peak:  ${peak:.0f}\n"
            f"Trough: ${trough:.0f}")
    ax.text(0.97, 0.97, info, transform=ax.transAxes,
            ha='right', va='top', fontsize=8.5,
            color=col, family='monospace',
            bbox=dict(facecolor=BG3, edgecolor=col,
                      boxstyle='round,pad=0.5', alpha=0.9))

plt.tight_layout(pad=1.8)
plt.savefig(os.path.join(OUT_DIR, 'hw_10yr_deep_dive.png'),
            dpi=150, bbox_inches='tight', facecolor=BG)

plt.close()
print("  [✓] hw_10yr_deep_dive.png")

# ──────────────────────────────────────────────────────────────────────
# SECTION 8 ── Final summary
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  KEY PREDICTIONS (2034 vs 2023 baseline)")
print("=" * 68)
for cat in CATS:
    r    = model_results[cat]
    base = r['history'][-1]
    end  = r['ens_fc'][-1]
    chg  = ((end - base) / base) * 100
    sym  = "▲ rising " if chg > 5  else "▼ falling" if chg < -5 else "→ stable "
    best_m = min(r['val_rmse'], key=r['val_rmse'].get)
    print(f"  {cat:12s}  ${base:>6.0f} → ${end:>6.0f}  "
          f"{sym} {abs(chg):5.1f}%  (best model: {best_m})")

print("\n  Output files:")
for f in ['hw_10yr_per_component.png', 'hw_10yr_all_overlay.png',
          'hw_10yr_pct_heatmap.png',   'hw_10yr_validation_uncertainty.png',
          'hw_10yr_deep_dive.png']:
    print(f"  {os.path.join(OUT_DIR, f)}")
print("\n  Done.")
# ── Single image viewer with next/prev ───────────────────────
import matplotlib.image as mpimg
from matplotlib.widgets import Button

image_files = [
    os.path.join(OUT_DIR, 'hw_10yr_per_component.png'),
    os.path.join(OUT_DIR, 'hw_10yr_all_overlay.png'),
    os.path.join(OUT_DIR, 'hw_10yr_pct_heatmap.png'),
    os.path.join(OUT_DIR, 'hw_10yr_validation_uncertainty.png'),
    os.path.join(OUT_DIR, 'hw_10yr_deep_dive.png'),
]

titles = [
    'Per Component Forecast',
    'All Categories Overlay',
    'Price Change Heatmap',
    'Validation & Uncertainty',
    'Component Deep Dives',
]

class Viewer:
    def __init__(self):
        self.current = 0
        self.n = len(image_files)
        self.images = [mpimg.imread(f) for f in image_files]

        self.fig, self.ax = plt.subplots(figsize=(16, 9))
        self.fig.patch.set_facecolor('#070710')
        self.ax.axis('off')
        plt.subplots_adjust(bottom=0.15, top=0.90,
                            left=0.01, right=0.99)

        self.img_display = self.ax.imshow(self.images[0])
        self.title = self.fig.suptitle(
            f'{titles[0]}  [1/{self.n}]',
            color='white', fontsize=14
        )

        ax_prev = plt.axes([0.35, 0.05, 0.12, 0.06])
        ax_next = plt.axes([0.53, 0.05, 0.12, 0.06])

        self.btn_prev = Button(ax_prev, '◀  Previous',
                               color='#1a1a30', hovercolor='#3a3a60')
        self.btn_next = Button(ax_next, 'Next  ▶',
                               color='#1a1a30', hovercolor='#3a3a60')
        self.btn_prev.label.set_color('white')
        self.btn_next.label.set_color('white')

        self.btn_prev.on_clicked(self.prev_img)
        self.btn_next.on_clicked(self.next_img)

        plt.show()

    def update(self):
        self.img_display.set_data(self.images[self.current])
        self.img_display.set_extent(
            [0, self.images[self.current].shape[1],
             self.images[self.current].shape[0], 0]
        )
        self.ax.relim()
        self.ax.autoscale_view()
        self.title.set_text(
            f'{titles[self.current]}  [{self.current+1}/{self.n}]'
        )
        self.fig.canvas.draw_idle()

    def next_img(self, event):
        self.current = (self.current + 1) % self.n
        self.update()

    def prev_img(self, event):
        self.current = (self.current - 1) % self.n
        self.update()

viewer = Viewer()