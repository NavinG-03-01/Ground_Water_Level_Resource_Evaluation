import numpy as np
import pandas as pd
from typing import Dict, List
from scipy import stats

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

def compute_linear_trend(series: pd.Series) -> Dict:
    x = np.arange(len(series))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, series.values)
    if p_value < 0.05:
        direction = "declining (depletion)" if slope > 0 else "recovering (recharge)"
    else:
        direction = "stationary (no significant trend)"
    return {
        "slope_m_per_day":  round(slope, 6),
        "slope_m_per_year": round(slope * 365, 4),
        "r_squared":        round(r_value ** 2, 4),
        "p_value":          round(p_value, 6),
        "significant":      bool(p_value < 0.05),
        "direction":        direction,
    }

def seasonal_decomposition(series: pd.Series, period: int = 12, model: str = "additive") -> Dict:
    if not HAS_STATSMODELS:
        return {"error": "statsmodels not installed"}
    if len(series) < 2 * period:
        period = max(2, len(series) // 2)
    try:
        result = seasonal_decompose(series, model=model, period=period, extrapolate_trend="freq")
        return {
            "model":              model,
            "period_months":      period,
            "seasonal_amplitude": round(float(result.seasonal.std() * 2), 3),
            "residual_std":       round(float(result.resid.std()), 4),
            "seasonal_values":    result.seasonal.round(4).tolist(),
            "trend_values":       result.trend.round(4).tolist(),
        }
    except Exception as e:
        return {"error": str(e)}

def monsoon_recharge_analysis(df: pd.DataFrame) -> Dict:
    df = df.copy()
    date_col = "date" if "date" in df.columns else "recorded_at"
    df["month"] = pd.to_datetime(df[date_col]).dt.month
    pre  = df[df["month"].isin([3, 4, 5])]["depth_to_water_m"]
    post = df[df["month"].isin([10, 11])]["depth_to_water_m"]
    if pre.empty or post.empty:
        return {"error": "Insufficient seasonal data"}
    recharge_m = float(pre.mean() - post.mean())
    return {
        "pre_monsoon_mean_m":  round(float(pre.mean()), 3),
        "post_monsoon_mean_m": round(float(post.mean()), 3),
        "recharge_m":          round(recharge_m, 3),
        "recharged":           recharge_m > 0,
        "recharge_status":     "Good Recharge" if recharge_m > 1 else
                               "Moderate Recharge" if recharge_m > 0 else "No Recharge / Deficit",
    }

def detect_anomalies_zscore(series: pd.Series, threshold: float = 3.0) -> List[Dict]:
    mean, std = series.mean(), series.std()
    if std == 0:
        return []
    z_scores = np.abs((series - mean) / std)
    return [
        {
            "index":   int(i),
            "value":   round(float(series.iloc[i]), 4),
            "z_score": round(float(z_scores.iloc[i]), 3),
        }
        for i in range(len(z_scores)) if z_scores.iloc[i] > threshold
    ]

def compute_summary_statistics(series: pd.Series) -> Dict:
    return {
        "count":  int(series.count()),
        "mean_m": round(float(series.mean()), 3),
        "std_m":  round(float(series.std()), 3),
        "min_m":  round(float(series.min()), 3),
        "max_m":  round(float(series.max()), 3),
        "p25_m":  round(float(series.quantile(0.25)), 3),
        "p50_m":  round(float(series.quantile(0.50)), 3),
        "p75_m":  round(float(series.quantile(0.75)), 3),
    }
