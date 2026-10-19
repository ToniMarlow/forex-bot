#!/usr/bin/env python3
"""
Бэктест №2: что из торговых идей вообще работает.

Каждая идея (тренд, MACD, RSI, свечи, паттерны, пробой, откат по тренду и
комбинация из бота) проверяется ОТДЕЛЬНО, с четырьмя вариантами выхода и
с учётом спреда. Плюс контроль: случайные входы (так выглядит «ноль»).
Чтобы не обмануть себя, история делится пополам: идея считается рабочей,
только если она в плюсе и в первой, и во второй половине.

Запуск:  python backtest2.py
"""
import numpy as np
import pandas as pd

import forex_analyzer as fa

WINDOW = 300
MIN_TRADES = 60
PIP = 0.0001
# Примерный спред у обычного брокера, в пунктах (поправь под своего)
SPREAD_PIPS = {"GBP/USD": 1.2, "EUR/USD": 0.9, "GBP/AUD": 2.8, "EUR/NZD": 3.5, "EUR/CAD": 2.5}
# (SL в ATR, TP в ATR или None, трейлинг-стоп)
EXITS = {
    "SL1.5/TP3": (1.5, 3.0, False),
    "SL1.5/TP1.5": (1.5, 1.5, False),
    "SL2/TP4": (2.0, 4.0, False),
    "трейлинг2": (2.0, None, True),
}
W = {k: fa.WEIGHTS[k] for k in ("technical", "candles", "patterns")}
WSUM = sum(W.values())


def ev(s):
    """Состояние -> событие: сигнал только в момент появления."""
    s = np.asarray(s, dtype=float)
    prev = np.r_[0, s[:-1]]
    return np.where((s != 0) & (s != prev), s, 0).astype(int)


def build(df):
    n = len(df)
    c, h, l = df["Close"], df["High"], df["Low"]
    e20, e50, e200 = fa.ema(c, 20), fa.ema(c, 50), fa.ema(c, 200)
    r = fa.rsi(c).values
    macd = fa.ema(c, 12) - fa.ema(c, 26)
    hist = (macd - fa.ema(macd, 9)).values
    trend = np.where((e20 > e50) & (e50 > e200), 1,
                     np.where((e20 < e50) & (e50 < e200), -1, 0))
    sig = {}
    sig["Тренд EMA20/50/200"] = ev(trend)
    sig["Цена выше/ниже EMA50"] = ev(np.where(c > e50, 1, -1))
    sig["MACD пересечение"] = ev(np.where(hist > 0, 1, -1))
    sig["RSI разворот (<30/>70)"] = ev(np.where(r < 30, 1, np.where(r > 70, -1, 0)))
    sig["Откат по тренду (RSI)"] = ev(np.where((trend == 1) & (r < 45), 1,
                                      np.where((trend == -1) & (r > 55), -1, 0)))
    hh, ll = h.rolling(20).max().shift(1), l.rolling(20).min().shift(1)
    sig["Пробой 20 свечей"] = ev(np.where(c > hh, 1, np.where(c < ll, -1, 0)))

    cd, pt, tt = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(WINDOW, n):
        win = df.iloc[i - WINDOW + 1:i + 1]
        cd[i] = fa.candle_score(win)[0]
        pt[i] = fa.pattern_score(win)[0]
        tt[i] = fa.technical_score(win)[0]
    sig["Свечные модели"] = ev(np.where(cd >= 33, 1, np.where(cd <= -33, -1, 0)))
    sig["Графические паттерны"] = ev(np.where(pt >= 30, 1, np.where(pt <= -30, -1, 0)))
    tot = (tt * W["technical"] + cd * W["candles"] + pt * W["patterns"]) / WSUM
    comb = np.where(tot >= fa.SIGNAL_THRESHOLD, 1, np.where(tot <= -fa.SIGNAL_THRESHOLD, -1, 0))
    comb = np.where((comb != 0) & (trend != 0) & (comb != trend) &
                    (np.abs(tot) < fa.COUNTER_TREND_THRESHOLD), 0, comb)
    sig["Комбинация (как в боте)"] = ev(comb)

    rng = np.random.default_rng(0)
    sig["Случайный вход (контроль)"] = np.where(rng.random(n) < 0.05, rng.choice([-1, 1], n), 0)
    return sig, fa.atr(df).values


def run(df, atr, evs, params, spread_price, start):
    sl_a, tp_a, trail = params
    maxhold = 40 if trail else 20
    c, hi, lo = df["Close"].values, df["High"].values, df["Low"].values
    n, out, i = len(c), [], start
    while i < n - 1:
        s = evs[i]
        if s == 0 or not np.isfinite(atr[i]) or atr[i] <= 0:
            i += 1
            continue
        entry, risk = c[i], sl_a * atr[i]
        sl = entry - s * risk
        tp = None if tp_a is None else entry + s * tp_a * atr[i]
        best, last, res, end = entry, min(i + maxhold, n - 1), None, None
        end = last
        for j in range(i + 1, last + 1):
            if (lo[j] <= sl) if s == 1 else (hi[j] >= sl):
                res, end = (sl - entry) * s / risk, j
                break
            if tp is not None and ((hi[j] >= tp) if s == 1 else (lo[j] <= tp)):
                res, end = (tp - entry) * s / risk, j
                break
            if trail:
                if s == 1:
                    best = max(best, hi[j]); sl = max(sl, best - risk)
                else:
                    best = min(best, lo[j]); sl = min(sl, best + risk)
        if res is None:
            res = (c[last] - entry) * s / risk
        out.append((i, res - spread_price / risk))
        i = end + 1
    return out


def summ(trades):
    if not trades:
        return 0, 0.0, 0.0
    r = np.array([t[2] for t in trades])
    return len(r), float(np.mean(r > 0) * 100), float(r.mean())


def main():
    data = {}
    for name, tick in fa.PAIRS.items():
        try:
            df = fa.fetch(tick, "1d", period="5y")
            sig, atr = build(df)
            data[name] = (df, sig, atr)
            print(f"{name}: {len(df)} свечей, сигналы посчитаны")
        except Exception as e:
            print(f"{name}: ошибка — {e}")
    if not data:
        return

    comps = list(next(iter(data.values()))[1].keys())
    mids = {nm: (WINDOW + len(d[0])) // 2 for nm, d in data.items()}
    res = {}  # (comp, exit) -> [(pair, half, R)]
    for comp in comps:
        for ex, params in EXITS.items():
            lst = []
            for nm, (df, sig, atr) in data.items():
                for i, r in run(df, atr, sig[comp], params, SPREAD_PIPS[nm] * PIP, WINDOW):
                    lst.append((nm, 0 if i < mids[nm] else 1, r))
            res[(comp, ex)] = lst

    print("\n=== Средний результат на сделку в R, после спреда (сделок) ===")
    print(f"{'идея':30s}" + "".join(f"{e:>18s}" for e in EXITS))
    for comp in comps:
        row = ""
        for ex in EXITS:
            n, _, avg = summ(res[(comp, ex)])
            row += f"{avg:>+10.2f}R ({n:>4d})"
        print(f"{comp:30s}{row}")

    print("\n=== Лучший выход для каждой идеи и проверка по половинам истории ===")
    print(f"{'идея':30s}{'выход':>13s}{'сделок':>8s}{'win%':>6s}{'всего':>8s}{'1-я':>8s}{'2-я':>8s}")
    cands = []
    for comp in comps:
        best = None
        for ex in EXITS:
            n, w, avg = summ(res[(comp, ex)])
            if n >= MIN_TRADES and (best is None or avg > best[3]):
                best = (ex, n, w, avg)
        if not best:
            print(f"{comp:30s}  мало сделок")
            continue
        ex, n, w, avg = best
        h1 = [t for t in res[(comp, ex)] if t[1] == 0]
        h2 = [t for t in res[(comp, ex)] if t[1] == 1]
        a1, a2 = summ(h1)[2], summ(h2)[2]
        ok = avg > 0.05 and a1 > 0 and a2 > 0 and "Случайный" not in comp
        if ok:
            cands.append((comp, ex))
        print(f"{comp:30s}{ex:>13s}{n:>8d}{w:>6.0f}{avg:>+8.2f}{a1:>+8.2f}{a2:>+8.2f}{'  <== кандидат' if ok else ''}")

    print("\n=== Вывод ===")
    if not cands:
        print("Ни одна идея не показала устойчивого плюса (в обеих половинах истории, после спреда).")
        print("Значит, на дневках по этим парам простые правила не дают преимущества.")
    else:
        print("Кандидаты (в плюсе в обеих половинах истории, после спреда):")
        for comp, ex in cands:
            print(f"\n{comp}, выход {ex}, по парам:")
            for nm in data:
                t = [x for x in res[(comp, ex)] if x[0] == nm]
                n, w, avg = summ(t)
                print(f"   {nm}: сделок {n:3d} | win {w:3.0f}% | {avg:+.2f}R")
        print("\nКандидат — это только повод проверить дальше (демо-счёт), а не гарантия.")
    print("Строка «Случайный вход» показывает уровень «ноль» с вашим спредом.")
    print("Прошлые результаты не гарантируют будущих.")


if __name__ == "__main__":
    main()
