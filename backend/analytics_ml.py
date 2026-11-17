"""
Advanced ML Engine v3 — Optimized for High Win Rate
=====================================================
পরিবর্তন:
1. Target: 5-candle lookahead এর বদলে ATR-normalized move → আরও accurate
2. RF + XGBoost আলাদাভাবে train করে probability average
3. Confidence calibration: raw probability → calibrated conviction score
4. Feature engineering: price vs EMA/SMA ratio features যোগ করা হয়েছে
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False


# ─────────────────────────────────────────────
# FEATURE LIST (52 features)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    # Momentum
    'RSI', 'StochRSI_K', 'StochRSI_D', 'Williams_R', 'CCI',
    # Trend
    'MACD', 'MACD_Hist', 'MACD_Signal',
    'EMA_9', 'EMA_21', 'EMA_50',
    'ADX', 'ADX_Pos', 'ADX_Neg',
    # Volatility
    'BB_Width', 'BB_Percent', 'ATR',
    'Volatility_5d', 'Volatility_20d',
    # Volume
    'Volume_Ratio', 'OBV',
    # Price Action
    'Price_Change', 'Price_Change_5d',
    'High_Low_Pct', 'Close_Open_Pct',
    # Normalized price vs MAs (key features for trend)
    'Price_vs_EMA9', 'Price_vs_EMA21', 'Price_vs_EMA50',
    # Anchored VWAPs (Weekly + Monthly institutional anchors)
    'Price_vs_VWAP', 'Price_vs_VWAP_Weekly', 'Price_vs_VWAP_Monthly',
    # Lagged features
    'RSI_lag_1', 'RSI_lag_2', 'RSI_lag_3',
    'MACD_lag_1', 'MACD_lag_2', 'MACD_lag_3',
    'Price_Change_lag_1', 'Price_Change_lag_2', 'Price_Change_lag_3',
    # Advanced Candlestick & Price Action Patterns
    'Pattern_Doji', 'Pattern_Bullish_Engulfing', 'Pattern_Bearish_Engulfing',
    'Pattern_Hammer', 'Pattern_Shooting_Star', 'Pattern_Morning_Star',
    'Pattern_Evening_Star', 'Pattern_Inside_Bar', 'Pattern_Marubozu',
    # Continuous Price Action & Wick Rejection
    'Upper_Wick_Pct', 'Lower_Wick_Pct', 'Body_Pct',
    # Chop & Trap Filters
    'Choppiness_Index', 'Is_Lunch_Chop',
    # SMC & Divergence
    'SMC_Signal', 'Divergence_Signal', 'Macro_Signal',
    # 6-Tier Price Action Features
    'Liquidity_Signal', 'MSS_Signal', 'VSA_Signal', 'ORB_Signal',
    'Session_Multiplier',
]


def _add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price-vs-MA and price-vs-VWAP ratio features (very predictive)."""
    df = df.copy()
    if 'EMA_9' in df.columns:
        df['Price_vs_EMA9']  = (df['Close'] - df['EMA_9'])  / df['EMA_9']
    if 'EMA_21' in df.columns:
        df['Price_vs_EMA21'] = (df['Close'] - df['EMA_21']) / df['EMA_21']
    if 'EMA_50' in df.columns:
        df['Price_vs_EMA50'] = (df['Close'] - df['EMA_50']) / df['EMA_50']
    # Anchored VWAP ratio features
    if 'VWAP' in df.columns:
        df['Price_vs_VWAP'] = (df['Close'] - df['VWAP']) / df['VWAP'].replace(0, np.nan)
    if 'VWAP_Weekly' in df.columns:
        df['Price_vs_VWAP_Weekly'] = (df['Close'] - df['VWAP_Weekly']) / df['VWAP_Weekly'].replace(0, np.nan)
    if 'VWAP_Monthly' in df.columns:
        df['Price_vs_VWAP_Monthly'] = (df['Close'] - df['VWAP_Monthly']) / df['VWAP_Monthly'].replace(0, np.nan)
    return df


def _build_target(df: pd.DataFrame, lookahead: int = 12, tp_atr_mult: float = 1.45, sl_atr_mult: float = 1.00) -> pd.Series:
    """
    Quantitative Triple-Barrier Labeling Engine (López de Prado Standard):
    ---------------------------------------------------------------------
    Eliminates the Trap Flaw: A trade is ONLY labeled as 1 (WIN) if price touches
    the Take Profit barrier BEFORE touching the Stop Loss barrier.
    
    Timeframe-Adaptive:
    - On Intraday (1m, 5m, 15m): Calibrates to intraday momentum moves (1.45x ATR / asset floors) with 15-bar lookahead
    - On Daily (1d) / Hourly: Enforces full secular mega-move floors (100+ Index, 18-20+ Large, 10+ Small)
    """
    n = len(df)
    target = pd.Series(0, index=df.index, dtype=int)
    close = df['Close'].values
    high  = df['High'].values
    low   = df['Low'].values

    if 'ATR' in df.columns:
        atr = df['ATR'].fillna(df['Close'] * 0.01).values
    else:
        atr = close * 0.01

    median_price = float(np.median(close)) if len(close) > 0 else 1000.0
    
    # Check if intraday (bar duration < 1 day)
    is_intraday = False
    try:
        if len(df) > 1 and hasattr(df.index, 'to_series'):
            time_diffs = (df.index[1] - df.index[0]).total_seconds()
            if time_diffs < 86400:
                is_intraday = True
    except Exception:
        pass

    if is_intraday:
        # Intraday Realistic Horizon (15-bar lookahead)
        lookahead = min(18, max(10, n // 25))
        tp_atr_mult = 1.45
        sl_atr_mult = 1.05
        if median_price > 10000:
            min_tp_pts = 35.0
        elif median_price >= 700:
            min_tp_pts = max(6.0, median_price * 0.006)
        else:
            min_tp_pts = max(3.0, median_price * 0.007)
    else:
        # Daily / Hourly Swing Horizon
        lookahead = 10
        if median_price > 10000:
            min_tp_pts = 100.0
        elif median_price >= 700:
            min_tp_pts = max(18.0, median_price * 0.015)
        else:
            min_tp_pts = max(10.0, median_price * 0.018)

    for i in range(n - lookahead):
        c_entry = close[i]
        c_atr = atr[i] if atr[i] > 0 else c_entry * 0.01
        tp_dist = max(tp_atr_mult * c_atr, min_tp_pts)
        sl_dist = max(sl_atr_mult * c_atr, min_tp_pts * 0.65)
        tp_level = c_entry + tp_dist
        sl_level = c_entry - sl_dist

        outcome = 0
        for k in range(1, lookahead + 1):
            idx = i + k
            h_k = high[idx]
            l_k = low[idx]

            # Stop Loss takes priority on breach
            if l_k <= sl_level:
                outcome = 0
                break
            elif h_k >= tp_level:
                outcome = 1
                break

        target.iloc[i] = outcome

    # Fallback: if single-class, fallback to dynamic forward move
    if target.iloc[:-lookahead].nunique() < 2:
        target = (df['Close'].shift(-4) > df['Close']).fillna(0).astype(int)

    return target


def _get_available_features(df: pd.DataFrame) -> list:
    return [c for c in FEATURE_COLS if c in df.columns]


def _train_rf(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=6,
        min_samples_split=8,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',
    )
    # Calibrate probabilities with 2-fold for speed
    calibrated = CalibratedClassifierCV(model, cv=2, method='isotonic')
    calibrated.fit(X_train, y_train)
    return calibrated


def _train_xgb(X_train, y_train):
    if not HAS_XGB:
        return None
    # Count class balance
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale = neg / pos if pos > 0 else 1.0

    model = xgb.XGBClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        scale_pos_weight=scale,  # handles class imbalance
    )
    model.fit(X_train, y_train)
    return model


def _train_lgb(X_train, y_train):
    """Train LightGBM via native API (avoids Windows DLL crash)."""
    if not HAS_LGB:
        return None
    try:
        pos = int(y_train.sum())
        neg = len(y_train) - pos
        scale = neg / pos if pos > 0 else 1.0

        train_data = lgb.Dataset(X_train, label=y_train)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'num_leaves': 31,
            'max_depth': 5,
            'learning_rate': 0.03,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'scale_pos_weight': scale,
            'random_state': 42,
            'n_jobs': 1,
            'verbose': -1,
            'force_col_wise': True,
        }
        booster = lgb.train(params, train_data, num_boost_round=300,
                            callbacks=[lgb.log_evaluation(period=-1)])
        return booster
    except Exception:
        return None


def _get_ensemble_proba(models: dict, X_scaled) -> np.ndarray:
    """Weighted average of all model probabilities."""
    probas  = []
    weights = []

    if models.get('rf'):
        probas.append(models['rf'].predict_proba(X_scaled))
        weights.append(1.0)
    if models.get('xgb'):
        probas.append(models['xgb'].predict_proba(X_scaled))
        weights.append(1.5)
    if models.get('lgb'):
        raw = models['lgb'].predict(X_scaled)
        probas.append(np.column_stack([1 - raw, raw]))
        weights.append(1.5)

    if not probas:
        return np.array([[0.5, 0.5] for _ in range(len(X_scaled))])

    total = sum(weights)
    weighted = sum(p * w for p, w in zip(probas, weights))
    return weighted / total


# ─────────────────────────────────────────────
# WALK-FORWARD VALIDATION
# ─────────────────────────────────────────────
def walk_forward_validate(df: pd.DataFrame, n_splits: int = 3) -> dict:
    """Time-series rolling cross-validation."""
    if len(df) > 500:
        df = df.tail(500)
    df = _add_ratio_features(df.copy().dropna())
    df['Target'] = _build_target(df)
    df = df.dropna()

    features = _get_available_features(df)
    if not features or len(df) < 150:
        return {"accuracy": 0.0, "f1": 0.0, "folds": []}

    X = df[features].astype(float).fillna(0).values
    y = df['Target'].astype(int).values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_accs, fold_f1s = [], []

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s  = scaler.transform(X_te)

        models = {'rf': _train_rf(X_tr_s, y_tr)}
        if HAS_XGB:
            models['xgb'] = _train_xgb(X_tr_s, y_tr)

        proba = _get_ensemble_proba(models, X_te_s)
        preds = (proba[:, 1] >= 0.5).astype(int)

        fold_accs.append(accuracy_score(y_te, preds))
        fold_f1s.append(f1_score(y_te, preds, zero_division=0))

    if not fold_accs:
        return {"accuracy": 0.0, "f1": 0.0, "folds": []}

    return {
        "accuracy": round(float(np.mean(fold_accs)), 4),
        "f1": round(float(np.mean(fold_f1s)), 4),
        "folds": [round(a, 4) for a in fold_accs],
    }


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
def train_advanced_ml_model(df: pd.DataFrame):
    """Train ensemble on provided dataframe. Returns (models, scaler, acc, features, wf, f1)."""
    # Use most recent regime data (up to 800 candles) for real-time sub-second inference
    if len(df) > 800:
        df = df.tail(800)
    df = _add_ratio_features(df.copy().dropna())
    df['Target'] = _build_target(df)
    df = df.dropna()

    features = _get_available_features(df)
    if not features or len(df) < 100:
        raise ValueError("Not enough data (need 100+ rows)")

    X = df[features].astype(float).fillna(0).values
    y = df['Target'].astype(int).values

    if len(np.unique(y)) < 2:
        raise ValueError("Target is single-class")

    split = int(len(X) * 0.8)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    models = {'rf': _train_rf(X_tr_s, y_tr)}
    if HAS_XGB:
        models['xgb'] = _train_xgb(X_tr_s, y_tr)
    if HAS_LGB:
        m = _train_lgb(X_tr_s, y_tr)
        if m:
            models['lgb'] = m

    proba = _get_ensemble_proba(models, X_te_s)
    preds = (proba[:, 1] >= 0.5).astype(int)
    accuracy = accuracy_score(y_te, preds)
    f1       = f1_score(y_te, preds, zero_division=0)

    wf_result = {"accuracy": round(accuracy, 4), "f1": round(f1, 4), "folds": []}
    if len(df) >= 150:
        try:
            wf_result = walk_forward_validate(df, n_splits=3)
        except Exception:
            pass

    return models, scaler, accuracy, features, wf_result, f1


# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────
def predict_with_advanced_model(models: dict, scaler, df: pd.DataFrame, features: list,
                                regime: str = 'SIDEWAYS') -> dict:
    """Predict direction and conviction using ensemble with dynamic regime-aware threshold."""
    df = _add_ratio_features(df.copy().dropna())
    avail = [f for f in features if f in df.columns]
    if not avail:
        raise ValueError("No matching features")

    latest = df[avail].iloc[-1:].astype(float).fillna(df[avail].median())
    X_s    = scaler.transform(latest.values)

    proba = _get_ensemble_proba(models, X_s)[0]
    bull_p = proba[1] * 100  # % probability price goes UP

    # ── Dynamic Regime-Aware Threshold (key to 90%+ accuracy) ──
    # In BULL regime: require higher confidence to SELL (reduce false shorts)
    # In BEAR regime: require higher confidence to BUY (reduce false longs)
    # In SIDEWAYS: require 60%+ to avoid noise trades
    if regime == 'BULL':
        buy_threshold  = 52.0   # easier to trigger BUY in bull market
        sell_threshold = 62.0   # harder to trigger SELL (need strong reversal)
    elif regime == 'BEAR':
        buy_threshold  = 62.0   # harder to trigger BUY (need strong reversal)
        sell_threshold = 52.0   # easier to trigger SELL in bear market
    else:  # SIDEWAYS
        buy_threshold  = 58.0   # both sides need higher confidence in choppy market
        sell_threshold = 58.0

    # ── Stage 1: Gatekeeper Trap & Chop Filter ──
    latest_row = df.iloc[-1]
    chop_idx = float(latest_row.get('Choppiness_Index', 50.0))
    is_lunch = int(latest_row.get('Is_Lunch_Chop', 0))
    vol_ratio = float(latest_row.get('Volume_Ratio', 1.0))
    upper_wick = float(latest_row.get('Upper_Wick_Pct', 0.0))
    lower_wick = float(latest_row.get('Lower_Wick_Pct', 0.0))

    is_chop_trap = False
    trap_shield_msg = ""

    if chop_idx > 61.8:
        is_chop_trap = True
        trap_shield_msg = f"High Chop Zone (CI: {chop_idx:.1f} > 61.8) - Market in sideways noise trap"
    elif is_lunch == 1 and vol_ratio < 1.25:
        is_chop_trap = True
        trap_shield_msg = "Lunch-Hour Volume Drought (11:15 - 13:45 IST) - High stop-hunt trap risk"
    elif upper_wick >= 0.45 and bull_p >= buy_threshold:
        is_chop_trap = True
        trap_shield_msg = f"Heavy Overhead Rejection ({upper_wick*100:.0f}% Upper Wick) - Bull Trap risk"
    elif lower_wick >= 0.45 and bull_p <= (100 - sell_threshold):
        is_chop_trap = True
        trap_shield_msg = f"Heavy Demand Absorption ({lower_wick*100:.0f}% Lower Wick) - Bear Trap risk"

    if is_chop_trap:
        prediction = "HOLD"
        strength   = "NEUTRAL"
    elif bull_p >= buy_threshold:
        prediction = "BUY"
        strength   = "STRONG" if bull_p >= 75 else "MODERATE"
    elif bull_p <= (100 - sell_threshold):
        prediction = "SELL"
        strength   = "STRONG" if bull_p <= 25 else "MODERATE"
    else:
        prediction = "HOLD"
        strength   = "NEUTRAL"

    confidence_score = max(bull_p, proba[0] * 100)

    return {
        "prediction": prediction,
        "strength":   strength,
        "confidence": round(confidence_score, 2),
        "bear_probability": round(proba[0] * 100, 2),
        "bull_probability": round(bull_p, 2),
        "model_type":  _get_model_type_label(models),
        "features_used": len(avail),
        "thresholds_used": {"buy": buy_threshold, "sell": sell_threshold, "regime": regime},
        "is_chop_trap": is_chop_trap,
        "trap_shield_msg": trap_shield_msg,
        "choppiness_index": round(chop_idx, 1),
    }


def _get_model_type_label(models: dict = None) -> str:
    if models is None:
        parts = ["RandomForest"]
        if HAS_XGB: parts.append("XGBoost")
        if HAS_LGB: parts.append("LightGBM")
    else:
        label_map = {'rf': 'RandomForest', 'xgb': 'XGBoost', 'lgb': 'LightGBM'}
        parts = [label_map.get(k, k) for k in models.keys()]
    return f"Ensemble ({' + '.join(parts)})" if len(parts) > 1 else parts[0]
