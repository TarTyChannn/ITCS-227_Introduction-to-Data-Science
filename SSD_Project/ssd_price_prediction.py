"""
=======================================================================
  SSD PRICE — 10-YEAR PRICE PREDICTION (2026–2036)
  Using Time Series Models: ARIMA, Holt's Double Exponential Smoothing,
  AR(p)+Seasonal, Ensemble — built on real Pangoly daily price data
  Libraries: numpy, scipy, sklearn, pandas, matplotlib
=======================================================================
"""

import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from scipy.signal import periodogram
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────
# SECTION 1 — Load & resample real SSD data to monthly
# ──────────────────────────────────────────────────────────────────────
CSV_PATH = 'ssd_prices_20260419_234829.csv'

print("=" * 68)
print("  SSD PRICE 10-YEAR PREDICTION  (2026 – 2036)")
print("=" * 68)
print("\n[1] Loading real SSD price data...")

df = pd.read_csv(CSV_PATH, parse_dates=['date'])
df = df.sort_values(['category', 'date'])

# Resample to monthly averages — daily data is too noisy for long forecasts
monthly = (
    df.groupby(['category', pd.Grouper(key='date', freq='MS')])['avg_price']
    .mean()
    .reset_index()
)

CATS = sorted(monthly['category'].unique())
print(f"  Categories : {len(CATS)}")
print(f"  Date range : {df['date'].min().date()} → {df['date'].max().date()}")
print(f"  Monthly obs: {monthly.groupby('category').size().to_dict()}\n")

# Build per-category monthly series dict
HISTORY = {}
for cat in CATS:
    sub = monthly[monthly['category'] == cat].set_index('date')['avg_price']
    sub = sub.dropna().sort_index()
    HISTORY[cat] = sub

FORECAST_MONTHS = 120   # 10 years
VAL_MONTHS      = 12    # hold-out validation window

# Forecast index starts from the month after last data point
last_date  = monthly['date'].max()
FC_INDEX   = pd.date_range(last_date + pd.offsets.MonthBegin(1),
                           periods=FORECAST_MONTHS, freq='MS')

print(f"  Forecast window: {FC_INDEX[0].date()} → {FC_INDEX[-1].date()}")

# ──────────────────────────────────────────────────────────────────────
# SECTION 2 — Real data statistics
# ──────────────────────────────────────────────────────────────────────
print("\n[2] Real dataset price statistics (monthly avg):")
print(f"  {'Category':22s}  {'N':>4}  {'Mean':>8}  {'Median':>8}  {'Std':>7}")
print("  " + "─" * 58)
for cat in CATS:
    s = HISTORY[cat]
    print(f"  {cat:22s}  {len(s):>4}  ${s.mean():>7.2f}  ${s.median():>7.2f}  ${s.std():>6.2f}")

# ──────────────────────────────────────────────────────────────────────
# SECTION 3 — Time Series Models
# ──────────────────────────────────────────────────────────────────────

def fit_arima_forecast(series, steps, p=4, d=1, q=2):
    y      = np.array(series, dtype=float)
    y_diff = np.diff(y, n=d)
    X_ar, Y_ar = [], []
    for i in range(p, len(y_diff)):
        X_ar.append(y_diff[i-p:i][::-1])
        Y_ar.append(y_diff[i])
    if len(X_ar) == 0:
        return np.full(steps, y[-1])
    X_ar, Y_ar = np.array(X_ar), np.array(Y_ar)
    ar_model   = Ridge(alpha=5.0)   # higher regularisation = less explosive
    ar_model.fit(X_ar, Y_ar)
    ar_fitted  = ar_model.predict(X_ar)
    residuals  = Y_ar - ar_fitted
    X_ma, Y_ma = [], []
    for i in range(q, len(residuals)):
        X_ma.append(residuals[i-q:i])
        Y_ma.append(residuals[i])
    ma_model = None
    if len(X_ma) > 0:
        ma_model = Ridge(alpha=5.0)
        ma_model.fit(np.array(X_ma), np.array(Y_ma))
    hist_diff = list(y_diff)
    hist_res  = list(residuals)
    preds_diff = []
    # Clamp each step's delta to ±3× the historical std of diffs
    max_delta = np.std(y_diff) * 3
    for _ in range(steps):
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
    result = list(y[-d:])
    for delta in preds_diff:
        result.append(result[-1] + delta)
    preds = np.array(result[d:])
    floor = y[-1] * 0.10          # floor at 10% of last known price
    ceil  = y[-1] * 15.0          # hard ceiling at 15× last known price
    return np.clip(preds, floor, ceil)


def select_arima_order(series, val_months=12):
    if len(series) < 2 * val_months + 10:
        return (4, 1, 2)
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


def fit_holts_forecast(series, steps, phi=0.88):
    y = np.array(series, dtype=float)
    def holts_sse(params):
        a, b = params
        if not (0 < a < 1 and 0 < b < 1):
            return 1e10
        L, T = y[0], y[1] - y[0]
        sse  = 0.0
        for i in range(1, len(y)):
            L_p, T_p = L, T
            L = a * y[i] + (1 - a) * (L_p + phi * T_p)
            T = b * (L - L_p) + (1 - b) * phi * T_p
            sse += (y[i] - (L_p + phi * T_p)) ** 2
        return sse
    best_sse, best_a, best_b = np.inf, 0.3, 0.1
    for a in np.arange(0.1, 0.91, 0.1):
        for b in np.arange(0.05, 0.41, 0.05):
            s = holts_sse([a, b])
            if s < best_sse:
                best_sse, best_a, best_b = s, a, b
    a, b = best_a, best_b
    L, T = y[0], y[1] - y[0]
    for i in range(1, len(y)):
        L_p, T_p = L, T
        L = a * y[i] + (1 - a) * (L_p + phi * T_p)
        T = b * (L - L_p) + (1 - b) * phi * T_p
    preds, phi_cumsum = [], 0.0
    for _ in range(steps):
        phi_cumsum = phi_cumsum * phi + phi
        preds.append(L + phi_cumsum * T)
    floor = y[-1] * 0.10
    ceil  = y[-1] * 15.0
    return np.clip(np.array(preds), floor, ceil), best_a, best_b


def fit_ar_seasonal_forecast(series, steps, p=12):
    y = np.array(series, dtype=float)
    n = len(y)
    if n <= p:
        return np.full(steps, y[-1])
    def make_row(i, hist):
        lag_feats = hist[i-p:i][::-1]
        t_norm    = i / n
        sine12 = np.sin(2 * np.pi * i / 12);  cos12 = np.cos(2 * np.pi * i / 12)
        sine24 = np.sin(2 * np.pi * i / 24);  cos24 = np.cos(2 * np.pi * i / 24)
        sine36 = np.sin(2 * np.pi * i / 36);  cos36 = np.cos(2 * np.pi * i / 36)
        sine60 = np.sin(2 * np.pi * i / 60);  cos60 = np.cos(2 * np.pi * i / 60)
        return list(lag_feats) + [t_norm,
                                   sine12, cos12,
                                   sine24, cos24,
                                   sine36, cos36,
                                   sine60, cos60]
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
    floor = y[-1] * 0.10
    ceil  = y[-1] * 15.0
    return np.clip(np.array(preds), floor, ceil)


def adaptive_ensemble(arima_fc, holts_fc, ar_fc, r_arima, r_holts, r_ar):
    eps = 1e-6
    inv = np.array([1 / (r_arima + eps),
                    1 / (r_holts  + eps),
                    1 / (r_ar     + eps)])
    w   = inv / inv.sum()
    fc  = w[0] * arima_fc + w[1] * holts_fc + w[2] * ar_fc
    return fc, w


def rmse(a, b):
    return np.sqrt(mean_squared_error(np.array(a), np.array(b)))

# ──────────────────────────────────────────────────────────────────────
# SECTION 4 — Run models per category
# ──────────────────────────────────────────────────────────────────────
print("\n[3] Running time series models & 10-year forecasts...\n")
print(f"  {'Category':22s}  {'ARIMA':>8}  {'Holts':>8}  {'AR':>8}  {'Ens':>8}  Weights(A/H/AR)")
print("  " + "─" * 78)

model_results = {}
np.random.seed(2026)

for cat in CATS:
    hist_series = HISTORY[cat].values
    hist_vals   = hist_series

    if len(hist_vals) < VAL_MONTHS + 10:
        print(f"  {cat:22s}  [SKIP — not enough data ({len(hist_vals)} months)]")
        continue

    train  = hist_vals[:-VAL_MONTHS]
    actual = hist_vals[-VAL_MONTHS:]

    best_ord      = select_arima_order(train, val_months=VAL_MONTHS)
    p_b, d_b, q_b = best_ord

    try:
        arima_val = fit_arima_forecast(train, VAL_MONTHS, p=p_b, d=d_b, q=q_b)
    except Exception:
        arima_val = np.full(VAL_MONTHS, train[-1])

    try:
        holts_val, a_opt, b_opt = fit_holts_forecast(train, VAL_MONTHS)
    except Exception:
        holts_val, a_opt, b_opt = np.full(VAL_MONTHS, train[-1]), 0.3, 0.1

    try:
        ar_val = fit_ar_seasonal_forecast(train, VAL_MONTHS, p=12)
    except Exception:
        ar_val = np.full(VAL_MONTHS, train[-1])

    r_arima = rmse(actual, arima_val)
    r_holts = rmse(actual, holts_val)
    r_ar    = rmse(actual, ar_val)

    ens_val, w_val = adaptive_ensemble(arima_val, holts_val, ar_val,
                                       r_arima, r_holts, r_ar)
    r_ens = rmse(actual, ens_val)

    print(f"  {cat:22s}  {r_arima:>7.2f}  {r_holts:>7.2f}  {r_ar:>7.2f}  "
          f"{r_ens:>7.2f}  {w_val[0]:.2f}/{w_val[1]:.2f}/{w_val[2]:.2f}")

    # Full forecast on all data
    try:
        arima_fc = fit_arima_forecast(hist_vals, FORECAST_MONTHS, p=p_b, d=d_b, q=q_b)
    except Exception:
        arima_fc = np.full(FORECAST_MONTHS, hist_vals[-1])

    try:
        holts_fc, _, _ = fit_holts_forecast(hist_vals, FORECAST_MONTHS)
    except Exception:
        holts_fc = np.full(FORECAST_MONTHS, hist_vals[-1])

    try:
        ar_fc = fit_ar_seasonal_forecast(hist_vals, FORECAST_MONTHS, p=12)
    except Exception:
        ar_fc = np.full(FORECAST_MONTHS, hist_vals[-1])

    ens_fc, w_full = adaptive_ensemble(arima_fc, holts_fc, ar_fc,
                                       r_arima, r_holts, r_ar)

    val_resid_std = np.std(actual - ens_val)
    horizon       = np.arange(1, FORECAST_MONTHS + 1)
    ci_width      = val_resid_std * np.sqrt(horizon / 4) * 1.96
    ci_lo_80      = ens_fc - val_resid_std * np.sqrt(horizon / 4) * 1.28
    ci_hi_80      = ens_fc + val_resid_std * np.sqrt(horizon / 4) * 1.28
    ci_lo_95      = ens_fc - ci_width
    ci_hi_95      = ens_fc + ci_width

    model_results[cat] = {
        'history':          hist_vals,
        'hist_index':       HISTORY[cat].index,
        'arima_fc':         arima_fc,
        'holts_fc':         holts_fc,
        'ar_fc':            ar_fc,
        'ens_fc':           ens_fc,
        'ci_lo_80':         np.clip(ci_lo_80, 0, None),
        'ci_hi_80':         ci_hi_80,
        'ci_lo_95':         np.clip(ci_lo_95, 0, None),
        'ci_hi_95':         ci_hi_95,
        'val_rmse':         {'ARIMA': r_arima, "Holt's": r_holts,
                             'AR+Seas': r_ar, 'Ensemble': r_ens},
        'best_arima_order': best_ord,
        'ens_weights':      w_full,
    }

CATS = list(model_results.keys())  # only cats with enough data

# ──────────────────────────────────────────────────────────────────────
# SECTION 5 — Forecast summary table
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  10-YEAR PRICE FORECAST SUMMARY (Ensemble)")
print("=" * 68)
# Year columns: 2026, 2028, 2030, 2032, 2034, 2036
yr_cols   = [2026, 2028, 2030, 2032, 2034, 2036]
yr_months = [(y - FC_INDEX[0].year) * 12 + (11 - FC_INDEX[0].month + 1)
             for y in yr_cols]
yr_months = [min(max(m, 0), FORECAST_MONTHS - 1) for m in yr_months]

print(f"  {'Category':22s}", end="")
for y in yr_cols:
    print(f" {y:>7}", end="")
print("  Trend")
print("  " + "─" * 72)

for cat in CATS:
    fc   = model_results[cat]['ens_fc']
    base = model_results[cat]['history'][-1]
    vals = [fc[min(m, len(fc)-1)] for m in yr_months]
    pct  = ((vals[-1] - base) / base) * 100
    sym  = "▲" if pct > 0 else "▼"
    print(f"  {cat:22s}", end="")
    for v in vals:
        print(f" ${v:>6.0f}", end="")
    print(f"  {sym}{abs(pct):.1f}%")

# ──────────────────────────────────────────────────────────────────────
# SECTION 6 — Spectral analysis
# ──────────────────────────────────────────────────────────────────────
print("\n[4] Dominant price cycles detected (periodogram):")
for cat in CATS:
    y = model_results[cat]['history']
    y_centered = y - y.mean()
    freqs, power = periodogram(y_centered, fs=1.0)
    valid = freqs > 0
    top3  = np.argsort(power[valid])[-3:][::-1]
    periods = 1 / freqs[valid][top3]
    print(f"  {cat:22s}: {', '.join([f'~{p:.0f}mo' for p in periods])}")

# ──────────────────────────────────────────────────────────────────────
# SECTION 7 — Plots
# ──────────────────────────────────────────────────────────────────────
print("\n[5] Generating plots...")

OUT_DIR = 'output'
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colours ──
BG     = '#070710'
BG2    = '#0e0e1c'
BG3    = '#13132a'
GRID_C = '#1a1a30'
WHITE  = '#e8e8ff'
DIM    = '#555588'

C_HIST  = '#3a7fff'
C_ARIMA = '#ff6644'
C_HOLTS = '#ffcc00'
C_AR    = '#44ffaa'
C_ENS   = '#ffffff'
C_CI80  = '#8888ff'
C_CI95  = '#5555cc'

# 21 distinct colours for categories
CAT_COLORS = [
    '#4a9eff','#ff5544','#ffcc00','#44ffaa','#ff88ff',
    '#ff9933','#88ccff','#ff44aa','#aaffdd','#ff6688',
    '#55ffff','#ffaa55','#aa88ff','#88ff88','#ff8844',
    '#44aaff','#ffff44','#ff4488','#44ff88','#8844ff',
    '#ff4444',
]
CAT_COLOR_MAP = {cat: CAT_COLORS[i % len(CAT_COLORS)] for i, cat in enumerate(CATS)}

dollar_fmt = FuncFormatter(lambda x, _: f'${x:.0f}')

def styled_ax(ax):
    ax.set_facecolor(BG2)
    ax.spines[['top','right','left','bottom']].set_color(GRID_C)
    ax.tick_params(colors=DIM, labelsize=8)
    ax.grid(True, color=GRID_C, lw=0.5, alpha=0.7)
    ax.xaxis.label.set_color(DIM)
    ax.yaxis.label.set_color(DIM)

# ══════════════════════════════════════════════════════════════
#  FIGURE 1 — Individual forecasts (grid: 5 cols × ceil rows)
# ══════════════════════════════════════════════════════════════
NCOLS = 4
NROWS = int(np.ceil(len(CATS) / NCOLS))
fig1  = plt.figure(figsize=(NCOLS * 7, NROWS * 5), facecolor=BG)
fig1.suptitle(
    'SSD Price — 10-Year Prediction (2026–2036)\n'
    "Models: ARIMA · Holt's ES · AR(12)+Seasonal · Weighted Ensemble",
    color=WHITE, fontsize=14, y=1.002, linespacing=1.6
)
gs1 = gridspec.GridSpec(NROWS, NCOLS, figure=fig1,
                        hspace=0.55, wspace=0.35)

for idx, cat in enumerate(CATS):
    ax  = fig1.add_subplot(gs1[idx // NCOLS, idx % NCOLS])
    styled_ax(ax)
    r   = model_results[cat]
    col = CAT_COLOR_MAP[cat]

    ax.plot(r['hist_index'], r['history'],
            color=C_HIST, lw=1.6, label='History', zorder=4)
    ax.plot(FC_INDEX, r['arima_fc'],
            color=C_ARIMA, lw=0.9, alpha=0.65, linestyle='--', label='ARIMA')
    ax.plot(FC_INDEX, r['holts_fc'],
            color=C_HOLTS, lw=0.9, alpha=0.65, linestyle=':', label="Holt's")
    ax.plot(FC_INDEX, r['ar_fc'],
            color=C_AR,    lw=0.9, alpha=0.65, linestyle='-.', label='AR+Seas')
    ax.fill_between(FC_INDEX, r['ci_lo_95'], r['ci_hi_95'],
                    color=C_CI95, alpha=0.18, label='95% CI')
    ax.fill_between(FC_INDEX, r['ci_lo_80'], r['ci_hi_80'],
                    color=C_CI80, alpha=0.28, label='80% CI')
    ax.plot(FC_INDEX, r['ens_fc'],
            color=C_ENS, lw=2.2, label='Ensemble', zorder=5)
    ax.axvline(FC_INDEX[0], color='#aa4488', lw=1.0,
               linestyle='--', alpha=0.7, zorder=3)

    base = r['history'][-1]
    end  = r['ens_fc'][-1]
    pct  = ((end - base) / base) * 100
    c    = '#ff4455' if pct > 0 else '#44ff88'
    sym  = '▲' if pct > 0 else '▼'
    ax.text(0.97, 0.04, f"{sym} {abs(pct):.1f}%\nby 2036",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=9, color=c, fontweight='bold',
            bbox=dict(facecolor=BG3, edgecolor=GRID_C,
                      boxstyle='round,pad=0.4', alpha=0.9))
    ax.set_title(cat, color=WHITE, fontsize=10, pad=6)
    ax.yaxis.set_major_formatter(dollar_fmt)
    ax.set_xlabel('Year', fontsize=7)
    ax.set_ylabel('Avg Price (USD)', fontsize=7)
    if idx == 0:
        ax.legend(fontsize=6.5, loc='upper right',
                  facecolor=BG3, edgecolor=GRID_C,
                  labelcolor=WHITE, framealpha=0.95)

# Hide unused subplots
for idx in range(len(CATS), NROWS * NCOLS):
    fig1.add_subplot(gs1[idx // NCOLS, idx % NCOLS]).set_visible(False)

plt.savefig(os.path.join(OUT_DIR, 'ssd_10yr_per_category.png'),
            dpi=140, bbox_inches='tight', facecolor=BG)
plt.close()
print("  [✓] ssd_10yr_per_category.png")

# ══════════════════════════════════════════════════════════════
#  FIGURE 2 — All categories overlay
# ══════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(22, 10), facecolor=BG)
styled_ax(ax2)
fig2.suptitle('All SSD Categories — Ensemble 10-Year Price Forecast (2026–2036)',
              color=WHITE, fontsize=14)

for cat in CATS:
    r   = model_results[cat]
    col = CAT_COLOR_MAP[cat]
    ax2.plot(r['hist_index'], r['history'],
             color=col, lw=1.0, alpha=0.3, linestyle=':')

ax2.axvline(FC_INDEX[0], color='#888899', lw=1.2,
            linestyle='--', alpha=0.6)
ax2.text(FC_INDEX[0], 1, '  ← History | Forecast →',
         color='#888899', fontsize=9, va='bottom',
         transform=ax2.get_xaxis_transform())

for yr in [2028, 2030, 2032, 2034, 2036]:
    ax2.axvline(pd.Timestamp(f'{yr}-01-01'), color=GRID_C, lw=0.7, alpha=0.5)

for cat in CATS:
    r   = model_results[cat]
    col = CAT_COLOR_MAP[cat]
    ax2.plot(FC_INDEX, r['ens_fc'], color=col, lw=2.0,
             label=cat, zorder=4)
    ax2.text(FC_INDEX[-1] + pd.offsets.MonthBegin(2),
             r['ens_fc'][-1],
             f' {cat}\n ${r["ens_fc"][-1]:.0f}',
             color=col, fontsize=6.5, va='center', fontweight='bold')

ax2.set_ylabel('Average Price (USD)', color=DIM, fontsize=10)
ax2.set_xlabel('Year', color=DIM, fontsize=10)
ax2.yaxis.set_major_formatter(dollar_fmt)
ax2.legend(fontsize=7, loc='upper left',
           facecolor=BG3, edgecolor=GRID_C, labelcolor=WHITE,
           ncol=3, framealpha=0.95)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'ssd_10yr_all_overlay.png'),
            dpi=140, bbox_inches='tight', facecolor=BG)
plt.close()
print("  [✓] ssd_10yr_all_overlay.png")

# ══════════════════════════════════════════════════════════════
#  FIGURE 3 — % Change heatmap
# ══════════════════════════════════════════════════════════════
years      = list(range(FC_INDEX[0].year, FC_INDEX[0].year + 11))
pct_matrix = np.zeros((len(years), len(CATS)))

for j, cat in enumerate(CATS):
    base = model_results[cat]['history'][-1]
    fc   = model_results[cat]['ens_fc']
    for i, yr in enumerate(years):
        mo_idx = min((yr - FC_INDEX[0].year) * 12 + 11, FORECAST_MONTHS - 1)
        pct_matrix[i, j] = ((fc[mo_idx] - base) / base) * 100

fig3, ax3 = plt.subplots(figsize=(20, 8), facecolor=BG)
ax3.set_facecolor(BG2)
fig3.suptitle('Predicted SSD Price Change (%) vs Baseline — Year × Category',
              color=WHITE, fontsize=13)

im = ax3.imshow(pct_matrix, cmap='RdYlGn_r', aspect='auto',
                vmin=-50, vmax=50)
ax3.set_xticks(range(len(CATS)))
ax3.set_xticklabels(CATS, color=WHITE, fontsize=8,
                    rotation=30, ha='right')
ax3.set_yticks(range(len(years)))
ax3.set_yticklabels([str(y) for y in years], color=WHITE, fontsize=9)

for i in range(len(years)):
    for j in range(len(CATS)):
        v   = pct_matrix[i, j]
        sym = '▲' if v > 0 else '▼'
        col = 'white' if abs(v) > 25 else '#111'
        ax3.text(j, i, f'{sym}{abs(v):.0f}%',
                 ha='center', va='center', fontsize=7,
                 color=col, fontweight='bold')

cbar = plt.colorbar(im, ax=ax3, shrink=0.85, pad=0.02)
cbar.ax.tick_params(colors=WHITE, labelsize=8)
cbar.set_label('Price Change (%)', color=WHITE, fontsize=9)
ax3.spines[['top','right','left','bottom']].set_color(GRID_C)
ax3.tick_params(colors=WHITE)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'ssd_10yr_pct_heatmap.png'),
            dpi=140, bbox_inches='tight', facecolor=BG)
plt.close()
print("  [✓] ssd_10yr_pct_heatmap.png")

# ══════════════════════════════════════════════════════════════
#  FIGURE 4 — Validation RMSE + forecast uncertainty
# ══════════════════════════════════════════════════════════════
fig4 = plt.figure(figsize=(22, 12), facecolor=BG)
gs4  = gridspec.GridSpec(2, 1, figure=fig4, hspace=0.55,
                          height_ratios=[1.4, 1])

ax_rmse = fig4.add_subplot(gs4[0])
styled_ax(ax_rmse)

model_names = ['ARIMA', "Holt's", 'AR+Seas', 'Ensemble']
m_colors    = [C_ARIMA, C_HOLTS, C_AR, C_ENS]
key_map     = {'ARIMA': 'ARIMA', "Holt's": "Holt's",
               'AR+Seas': 'AR+Seas', 'Ensemble': 'Ensemble'}
n_cats = len(CATS); n_mod = len(model_names)
x      = np.arange(n_cats)
bar_w  = 0.18

for mi, (mname, mcol) in enumerate(zip(model_names, m_colors)):
    rmse_vals = [model_results[c]['val_rmse'][key_map[mname]] for c in CATS]
    bars = ax_rmse.bar(x + mi * bar_w - bar_w * 1.5, rmse_vals,
                       bar_w, color=mcol, alpha=0.82, label=mname,
                       edgecolor=BG, linewidth=0.4)
    for bar, v in zip(bars, rmse_vals):
        ax_rmse.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.1, f'{v:.1f}',
                     ha='center', va='bottom', fontsize=6,
                     color=mcol, alpha=0.9)

ax_rmse.set_xticks(x)
ax_rmse.set_xticklabels(CATS, color=WHITE, fontsize=7,
                         rotation=25, ha='right')
ax_rmse.set_ylabel('RMSE (USD)', color=DIM, fontsize=10)
ax_rmse.set_title('12-Month Validation RMSE — All Models × All SSD Categories',
                  color=WHITE, fontsize=11, pad=8)
ax_rmse.legend(fontsize=9, facecolor=BG3, edgecolor=GRID_C,
               labelcolor=WHITE, loc='upper right')

ax_unc = fig4.add_subplot(gs4[1])
styled_ax(ax_unc)

horizon_labels = [str(y) for y in yr_cols]
x_pos = np.arange(len(horizon_labels))
cat_gap = 0.06

for ci, cat in enumerate(CATS):
    r    = model_results[cat]
    meds = np.array([r['ens_fc'][min(m, FORECAST_MONTHS-1)] for m in yr_months])
    lo80 = np.array([r['ci_lo_80'][min(m, FORECAST_MONTHS-1)] for m in yr_months])
    hi80 = np.array([r['ci_hi_80'][min(m, FORECAST_MONTHS-1)] for m in yr_months])
    x_off = x_pos + ci * cat_gap - (n_cats / 2) * cat_gap
    col = CAT_COLOR_MAP[cat]
    ax_unc.plot(x_off, meds, 'o-', color=col, lw=1.2, ms=3,
                label=cat, alpha=0.85, zorder=3)
    ax_unc.fill_between(x_off, lo80, hi80, color=col, alpha=0.08)

ax_unc.set_xticks(x_pos)
ax_unc.set_xticklabels(horizon_labels, color=WHITE, fontsize=10)
ax_unc.set_ylabel('Forecast Price (USD)', color=DIM, fontsize=10)
ax_unc.set_title('Ensemble Forecast Trajectory + 80% Confidence Interval',
                 color=WHITE, fontsize=11, pad=8)
ax_unc.yaxis.set_major_formatter(dollar_fmt)
ax_unc.legend(fontsize=7, facecolor=BG3, edgecolor=GRID_C,
              labelcolor=WHITE, ncol=4, loc='upper right')

plt.savefig(os.path.join(OUT_DIR, 'ssd_10yr_validation_uncertainty.png'),
            dpi=140, bbox_inches='tight', facecolor=BG)
plt.close()
print("  [✓] ssd_10yr_validation_uncertainty.png")

# ══════════════════════════════════════════════════════════════
#  FIGURE 5 — Deep dive bar charts per category
# ══════════════════════════════════════════════════════════════
fig5 = plt.figure(figsize=(NCOLS * 7, NROWS * 5), facecolor=BG)
fig5.suptitle('SSD Price Predictions — Category Deep Dives (2026–2036)',
              color=WHITE, fontsize=15, y=1.01)
gs5 = gridspec.GridSpec(NROWS, NCOLS, figure=fig5,
                         hspace=0.65, wspace=0.4)

for idx, cat in enumerate(CATS):
    ax  = fig5.add_subplot(gs5[idx // NCOLS, idx % NCOLS])
    ax.set_facecolor(BG2)
    styled_ax(ax)
    r   = model_results[cat]
    fc  = r['ens_fc']
    col = CAT_COLOR_MAP[cat]

    ann_n    = min(10, FORECAST_MONTHS // 12)
    ann_idx  = [i * 12 + 11 for i in range(ann_n)]
    ann_fc   = fc[ann_idx]
    ann_lo   = r['ci_lo_80'][ann_idx]
    ann_hi   = r['ci_hi_80'][ann_idx]
    ann_yrs  = [FC_INDEX[0].year + i for i in range(ann_n)]

    ax.bar(ann_yrs, ann_fc, color=col, alpha=0.45,
           edgecolor=col, linewidth=0.8, width=0.7)
    ax.errorbar(ann_yrs, ann_fc,
                yerr=[ann_fc - ann_lo, ann_hi - ann_fc],
                fmt='none', ecolor=col, elinewidth=1.2,
                capsize=4, capthick=1.2, alpha=0.7)
    z = np.polyfit(ann_yrs, ann_fc, 1)
    p = np.poly1d(z)
    x_l = np.linspace(ann_yrs[0], ann_yrs[-1], 100)
    ax.plot(x_l, p(x_l), color=WHITE, lw=1.2,
            linestyle='--', alpha=0.5)

    base      = r['history'][-1]
    total_chg = ((fc[-1] - base) / base) * 100
    arr_col   = '#ff4455' if total_chg > 0 else '#44ff88'
    arr_sym   = '▲' if total_chg > 0 else '▼'

    ax.set_title(cat, color=WHITE, fontsize=9, pad=5, fontweight='bold')
    ax.set_ylabel('Price (USD)', color=DIM, fontsize=7)
    ax.yaxis.set_major_formatter(dollar_fmt)
    ax.tick_params(axis='x', rotation=35, labelsize=7)

    info = (f"Base:  ${base:.2f}\n"
            f"2036:  ${fc[-1]:.2f}\n"
            f"Chg:   {arr_sym}{abs(total_chg):.1f}%\n"
            f"Peak:  ${fc.max():.2f}\n"
            f"Low:   ${fc.min():.2f}")
    ax.text(0.97, 0.97, info, transform=ax.transAxes,
            ha='right', va='top', fontsize=7.5,
            color=col, family='monospace',
            bbox=dict(facecolor=BG3, edgecolor=col,
                      boxstyle='round,pad=0.4', alpha=0.9))

for idx in range(len(CATS), NROWS * NCOLS):
    fig5.add_subplot(gs5[idx // NCOLS, idx % NCOLS]).set_visible(False)

plt.tight_layout(pad=1.5)
plt.savefig(os.path.join(OUT_DIR, 'ssd_10yr_deep_dive.png'),
            dpi=140, bbox_inches='tight', facecolor=BG)
plt.close()
print("  [✓] ssd_10yr_deep_dive.png")

# ──────────────────────────────────────────────────────────────────────
# SECTION 8 — Final summary
# ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  KEY PREDICTIONS (2036 vs 2026 baseline)")
print("=" * 68)
for cat in CATS:
    r    = model_results[cat]
    base = r['history'][-1]
    end  = r['ens_fc'][-1]
    chg  = ((end - base) / base) * 100
    sym  = "▲ rising " if chg > 5  else "▼ falling" if chg < -5 else "→ stable "
    best_m = min(r['val_rmse'], key=r['val_rmse'].get)
    print(f"  {cat:22s}  ${base:>7.2f} → ${end:>7.2f}  "
          f"{sym} {abs(chg):5.1f}%  (best: {best_m})")

print("\n  Output files saved to:", OUT_DIR)
for f in ['ssd_10yr_per_category.png', 'ssd_10yr_all_overlay.png',
          'ssd_10yr_pct_heatmap.png', 'ssd_10yr_validation_uncertainty.png',
          'ssd_10yr_deep_dive.png']:
    print(f"  → {f}")
print("\n  Done.")
