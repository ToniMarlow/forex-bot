#!/usr/bin/env python3
"""
Интрадей-движок: решительные сигналы на 1-часовых свечах, сделка живёт до 4 часов.

Один и тот же код используется и в боте, и в бэктесте (backtest_intraday.py),
поэтому то, что проверено на истории, и есть то, что приходит в Telegram.

Сигнал выдаётся, только если одновременно выполнено ВСЁ:
  1) сработало одно из правил на закрытой 1H-свече (EMA50 / Stochastic 20-80 / RSI 30-70);
  2) направление совпадает с трендом 4H (EMA50 и EMA200 на 4H);
  3) на 4H есть тренд, а не боковик (ADX >= ADX_MIN);
  4) время активной сессии (SESSION, UTC);
  5) в ближайшие часы нет важных новостей (проверяет бот).
Класс сигнала (A/B/C) зависит от числа дополнительных подтверждений.

Это инструмент поддержки решений, а не гарантия прибыли.
"""
import datetime as dt

import numpy as np
import pandas as pd

import forex_analyzer as fa

# ------------------------- НАСТРОЙКИ -------------------------
HOLD_BARS = 4                       # максимум 4 часовые свечи = 4 часа
EXITS = {"A": (1.0, 1.5),           # (SL, TP) в ATR 1H
         "B": (1.5, 2.0)}
EXIT_KEY = "A"                      # какой выход использует бот (после бэктеста можно сменить)
SESSION = (7, 16)                   # входы только в 07:00-16:00 UTC (Лондон + начало Нью-Йорка)
ADX_MIN = 20.0
MIN_PTS = 2                         # минимум доп. подтверждений для отправки (2 = класс B и выше)
NEWS_BLOCK_HOURS = 2                # не давать сигнал, если важная новость ближе N часов
FRESH_MINUTES = 30                  # сигнал живёт после закрытия свечи не дольше N минут
PIP = 0.0001
SPREAD_PIPS = {"GBP/USD": 1.2, "EUR/USD": 0.9, "GBP/AUD": 2.8, "EUR/NZD": 3.5, "EUR/CAD": 2.5}
RULES = ("ema50", "stoch", "rsi")

# Сюда после запуска backtest_intraday.py вписывается короткий итог проверки на истории.
# Пока пусто, бот честно пишет, что бэктест не подтверждён.
BT_NOTE = ""

WHY = {
    ("ema50", 1): "цена закрылась выше EMA50: откат закончился, движение вверх продолжается",
    ("ema50", -1): "цена закрылась ниже EMA50: отскок закончился, движение вниз продолжается",
    ("stoch", 1): "Stochastic вошёл в зону перепроданности (ниже 20): откат в тренде вверх выкуплен",
    ("stoch", -1): "Stochastic вошёл в зону перекупленности (выше 80): отскок в тренде вниз продан",
    ("rsi", 1): "RSI упал ниже 30: сильный откат в тренде вверх, ждём возврат",
    ("rsi", -1): "RSI поднялся выше 70: сильный отскок в тренде вниз, ждём возврат",
}


# ------------------------- ДАННЫЕ -------------------------
def to_utc(df):
    idx = df.index
    df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return df


def fetch_1h(ticker, period="730d"):
    df = fa.yf.download(ticker, period=period, interval="1h", progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError("нет данных")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    return to_utc(df)


# ------------------------- ИНДИКАТОРЫ -------------------------
def adx(df, n=14):
    h, l = df["High"], df["Low"]
    up, dn = h.diff(), -l.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = fa.atr(df, n)
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def htf(df):
    """Тренд и ADX на 4H, видимые на 1H без заглядывания в будущее (только завершённые 4H-свечи)."""
    h4 = df.resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    c = h4["Close"]
    e50, e200 = fa.ema(c, 50), fa.ema(c, 200)
    trend = pd.Series(np.where((c > e50) & (e50 > e200), 1, np.where((c < e50) & (e50 < e200), -1, 0)),
                      index=h4.index)
    f = pd.DataFrame({"trend": trend, "adx": adx(h4)}).shift(1)
    return f.reindex(df.index, method="ffill")


def events(state):
    st = np.asarray(state, dtype=int)
    prev = np.r_[0, st[:-1]]
    return np.where((st != 0) & (st != prev), st, 0)


def build(df):
    c, h, l = df["Close"], df["High"], df["Low"]
    e50, e200 = fa.ema(c, 50), fa.ema(c, 200)
    r = fa.rsi(c).values
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    k = (100 * (c - ll) / (hh - ll).replace(0, np.nan)).rolling(3).mean().values
    states = {
        "ema50": np.sign((c - e50).values),
        "stoch": np.where(k < 20, 1, np.where(k > 80, -1, 0)),
        "rsi": np.where(r < 30, 1, np.where(r > 70, -1, 0)),
    }
    ev = {name: events(s) for name, s in states.items()}
    atr = fa.atr(df).values
    f = htf(df)
    aux = dict(e200=e200.values, rsi=r, atr=atr,
               atr_med=pd.Series(atr).rolling(200, min_periods=50).median().values,
               trend=f["trend"].values, adx=f["adx"].values)
    return ev, aux


# ------------------------- ФИЛЬТРЫ И КЛАСС СИГНАЛА -------------------------
def hard_ok(df, aux, d, flt):
    """Жёсткие фильтры: сигнал допустим только если все включённые условия выполнены."""
    ok = np.ones(len(df), bool)
    if "session" in flt:
        hr, wd = df.index.hour.values, df.index.weekday.values
        ok &= (hr >= SESSION[0]) & (hr < SESSION[1]) & (wd < 5)
    if "trend" in flt:
        ok &= aux["trend"] == d
    if "adx" in flt:
        ok &= aux["adx"] >= ADX_MIN
    return ok


def grade_points(aux, ev, d, close):
    """Доп. подтверждения (0..4): совпадение правил, сторона EMA200, RSI не в экстремуме, нормальная волатильность."""
    d = np.asarray(d)
    cnt_pos = np.zeros(len(d))
    cnt_neg = np.zeros(len(d))
    for r in RULES:
        cnt_pos += pd.Series(ev[r] == 1).rolling(3, min_periods=1).max().values
        cnt_neg += pd.Series(ev[r] == -1).rolling(3, min_periods=1).max().values
    confl = np.where(d == 1, cnt_pos, cnt_neg) >= 2
    side = ((close > aux["e200"]) & (d == 1)) | ((close < aux["e200"]) & (d == -1))
    rsi_ok = ((d == 1) & (aux["rsi"] < 70)) | ((d == -1) & (aux["rsi"] > 30))
    vol_ok = (aux["atr"] >= 0.6 * aux["atr_med"]) & (aux["atr"] <= 2.0 * aux["atr_med"])
    return confl.astype(int) + side.astype(int) + rsi_ok.astype(int) + vol_ok.astype(int)


def grade_letter(pts):
    return "A" if pts >= 3 else "B" if pts == 2 else "C"


# ------------------------- СИМУЛЯЦИЯ СДЕЛКИ (для бэктеста) -------------------------
def sim(df, atr, i, d, sl_a, tp_a, spread):
    n = len(df)
    j_end = i + HOLD_BARS
    if j_end >= n:
        return None
    if df.index[j_end] - df.index[i] > pd.Timedelta(hours=HOLD_BARS + 1):
        return None     # пропуск в данных или выходные: сделку не считаем
    c, h, l = df["Close"].values, df["High"].values, df["Low"].values
    a = atr[i]
    if not np.isfinite(a) or a <= 0:
        return None
    entry, risk = c[i], sl_a * a
    sl, tp = entry - d * risk, entry + d * tp_a * a
    for j in range(i + 1, j_end + 1):
        sl_hit = (l[j] <= sl) if d == 1 else (h[j] >= sl)
        tp_hit = (h[j] >= tp) if d == 1 else (l[j] <= tp)
        if sl_hit:                          # если в одной свече и SL, и TP, считаем по худшему
            return -1.0 - spread / risk, j
        if tp_hit:
            return tp_a / sl_a - spread / risk, j
    return (c[j_end] - entry) * d / risk - spread / risk, j_end


def trades_for(df, aux, ev, rule, flt, exit_key, spread, start=250, seed=0):
    """Список (индекс, результат в R, очки подтверждений). Сделки одной пары не пересекаются."""
    sl_a, tp_a = EXITS[exit_key]
    n = len(df)
    if rule == "rand":
        rng = np.random.default_rng(seed)
        d_arr = np.where(rng.random(n) < 0.03, rng.choice([-1, 1], n), 0)
    else:
        d_arr = ev[rule]
    ok = hard_ok(df, aux, d_arr, flt)
    pts = grade_points(aux, ev, d_arr, df["Close"].values)
    out, busy = [], -1
    for i in np.flatnonzero((d_arr != 0) & ok):
        if i < start or i <= busy:
            continue
        res = sim(df, aux["atr"], i, int(d_arr[i]), sl_a, tp_a, spread)
        if res is None:
            continue
        r, j = res
        out.append((int(i), float(r), int(pts[i])))
        busy = j
    return out


# ------------------------- ЖИВОЙ СИГНАЛ -------------------------
def latest_signal(pair, df=None, now=None):
    """(сигнал или None, свежие_ли_данные). Смотрит последнюю ЗАКРЫТУЮ часовую свечу."""
    now = now or pd.Timestamp.now(tz="UTC")
    if df is None:
        df = fetch_1h(fa.PAIRS[pair], "120d")
    if df.index[-1] + pd.Timedelta(hours=1) > now:      # последняя свеча ещё не закрыта
        df = df.iloc[:-1]
    expected = now.floor("h") - pd.Timedelta(hours=1)
    fresh = df.index[-1] == expected
    if not fresh or len(df) < 260:
        return None, fresh
    ev, aux = build(df)
    i = len(df) - 1
    cands = {r: int(ev[r][i]) for r in RULES if ev[r][i] != 0}
    if not cands or len(set(cands.values())) > 1:
        return None, True
    d = next(iter(cands.values()))
    darr = np.full(len(df), d)
    if not hard_ok(df, aux, darr, {"session", "trend", "adx"})[i]:
        return None, True
    c = df["Close"].values
    pts = int(grade_points(aux, ev, darr, c)[i])
    a = float(aux["atr"][i])
    if not np.isfinite(a) or a <= 0:
        return None, True
    sl_a, tp_a = EXITS[EXIT_KEY]
    entry = float(c[i])
    bar_end = df.index[i] + pd.Timedelta(hours=1)
    why = [WHY[(r, d)] for r in cands]
    why.append(f"тренд 4H {'вверх' if d == 1 else 'вниз'} (EMA50 {'>' if d == 1 else '<'} EMA200), "
               f"ADX 4H {aux['adx'][i]:.0f}, активная сессия")
    return dict(pair=pair, dir=d, pts=pts, grade=grade_letter(pts), rules=list(cands),
                bar_ts=df.index[i].isoformat(), bar_end=bar_end.timestamp(),
                expires=(bar_end + pd.Timedelta(hours=HOLD_BARS)).timestamp(),
                entry=entry, sl=entry - d * sl_a * a, tp=entry + d * tp_a * a, atr=a, why=why), True


def explain(pair, now=None, lookback=24):
    """Диагностика: почему нет сигнала. Считает, какие фильтры отсекли события за последние lookback часов."""
    now = now or pd.Timestamp.now(tz="UTC")
    df = fetch_1h(fa.PAIRS[pair], "120d")
    if df.index[-1] + pd.Timedelta(hours=1) > now:
        df = df.iloc[:-1]
    fresh = df.index[-1] == now.floor("h") - pd.Timedelta(hours=1)
    ev, aux = build(df)
    n = len(df)
    dd = np.zeros(n, int)
    conflict = np.zeros(n, bool)
    for j in range(n - lookback, n):
        dirs = {int(ev[r][j]) for r in RULES if ev[r][j] != 0}
        if len(dirs) == 1:
            dd[j] = next(iter(dirs))
        conflict[j] = len(dirs) > 1
    pts = grade_points(aux, ev, dd, df["Close"].values)
    s_ok = hard_ok(df, aux, dd, {"session"})
    res = dict(pair=pair, bar=df.index[-1], fresh=fresh, trend=aux["trend"][-1], adx=aux["adx"][-1],
               events=0, session=0, trend_f=0, adx_f=0, weak=0, passed=0, conflict=int(conflict.sum()))
    for j in range(n - lookback, n):
        if dd[j] == 0:
            continue
        res["events"] += 1
        if not s_ok[j]:
            res["session"] += 1
        elif aux["trend"][j] != dd[j]:
            res["trend_f"] += 1
        elif not aux["adx"][j] >= ADX_MIN:
            res["adx_f"] += 1
        elif pts[j] < MIN_PTS:
            res["weak"] += 1
        else:
            res["passed"] += 1
    return res
