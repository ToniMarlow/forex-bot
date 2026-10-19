#!/usr/bin/env python3
"""
Бэктест: проверяем на истории (дневки, ~5 лет), насколько работают сигналы.

Учитываются только технический анализ, свечи и графические паттерны
(фундаментал и новости за прошлые годы бесплатно не достать).
Сделка: вход по цене закрытия сигнальной свечи, SL = 1.5 ATR, TP = 3 ATR
(риск:прибыль 1:2), максимум 20 свечей в сделке. Если SL и TP задеты
в одной свече, считаем, что сработал SL (консервативно).

Запуск:  python backtest.py
"""
import numpy as np
import pandas as pd

import forex_analyzer as fa

WINDOW = 300          # сколько свечей смотрим на каждом шаге
MAX_HOLD = 20         # максимум свечей в сделке
THRESHOLDS = [20, 25, 30, 40, 50]
W = {k: fa.WEIGHTS[k] for k in ("technical", "candles", "patterns")}
WSUM = sum(W.values())


def hist_signal(win, threshold, use_trend_filter):
    t, _ = fa.technical_score(win)
    cd, _ = fa.candle_score(win)
    pt, _ = fa.pattern_score(win)
    total = (t * W["technical"] + cd * W["candles"] + pt * W["patterns"]) / WSUM
    if total >= threshold:
        sig = 1
    elif total <= -threshold:
        sig = -1
    else:
        return 0, total
    if use_trend_filter:
        tr = fa.trend_dir(win)
        if tr != 0 and sig != tr and abs(total) < fa.COUNTER_TREND_THRESHOLD:
            return 0, total
    return sig, total


def simulate(df, i, sig, a):
    """Возвращает результат сделки в R (риск = 1)."""
    entry = df["Close"].iloc[i]
    risk = 1.5 * a
    sl = entry - sig * risk
    tp = entry + sig * 2 * risk
    last = min(i + MAX_HOLD, len(df) - 1)
    for j in range(i + 1, last + 1):
        hi, lo = df["High"].iloc[j], df["Low"].iloc[j]
        hit_sl = lo <= sl if sig == 1 else hi >= sl
        hit_tp = hi >= tp if sig == 1 else lo <= tp
        if hit_sl:
            return -1.0, j
        if hit_tp:
            return 2.0, j
    return float((df["Close"].iloc[last] - entry) * sig / risk), last


def run_pair(df, threshold, use_trend_filter):
    atr_s = fa.atr(df)
    res, i = [], WINDOW
    while i < len(df) - 1:
        win = df.iloc[i - WINDOW + 1:i + 1]
        sig, total = hist_signal(win, threshold, use_trend_filter)
        if sig == 0:
            i += 1
            continue
        r, end = simulate(df, i, sig, atr_s.iloc[i])
        res.append(r)
        i = end + 1
    return res


def stats(res):
    if not res:
        return "сделок нет"
    r = np.array(res)
    return (f"сделок {len(r):3d} | прибыльных {np.mean(r > 0) * 100:4.0f}% | "
            f"средний результат {r.mean():+.2f}R | всего {r.sum():+.1f}R")


def main():
    data = {}
    for name, tick in fa.PAIRS.items():
        try:
            data[name] = fa.fetch(tick, "1d", period="5y")
            print(f"{name}: {len(data[name])} свечей")
        except Exception as e:
            print(f"{name}: ошибка — {e}")

    print("\n=== Подбор порога (все пары вместе, с фильтром по тренду) ===")
    for th in THRESHOLDS:
        allr = []
        for name, df in data.items():
            allr += run_pair(df, th, True)
        print(f"порог {th:2d}: {stats(allr)}")

    th = fa.SIGNAL_THRESHOLD
    print(f"\n=== Фильтр по тренду при пороге {th} ===")
    for flag in (False, True):
        allr = []
        for name, df in data.items():
            allr += run_pair(df, th, flag)
        print(f"{'с фильтром  ' if flag else 'без фильтра '}: {stats(allr)}")

    print(f"\n=== По парам (порог {th}, с фильтром) ===")
    for name, df in data.items():
        print(f"{name}: {stats(run_pair(df, th, True))}")

    print("\nСредний результат > 0 — у стратегии есть преимущество на истории;"
          " около 0 или ниже — сигналы не лучше монетки.")
    print("Прошлые результаты не гарантируют будущих.")


if __name__ == "__main__":
    main()
