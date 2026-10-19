#!/usr/bin/env python3
"""
Бэктест №3: можно ли поднять процент прибыльных сделок.

Берём три правила бота и пробуем добавлять к ним фильтры (тренд, сила тренда,
недельный тренд, подтверждение другим правилом и т.д.). Фильтр полезен, только
если он улучшает И процент прибыльных, И средний результат после спреда,
и это держится в обеих половинах истории.

Вторая таблица показывает обмен «процент прибыльных <-> размер прибыли»:
близкий TP даёт больше прибыльных сделок, но каждая приносит меньше.

Запуск:  python backtest3.py
"""
import numpy as np
import pandas as pd

import backtest2 as b2
import forex_analyzer as fa

START = 250
MIN_N = 40
RULES = {   # правило -> (название, выход SL/TP в ATR)
    "ema50": ("Цена пересекла EMA50", (2.0, 4.0, False)),
    "macd": ("Пересечение MACD", (1.5, 3.0, False)),
    "rsi": ("Разворот RSI 30/70", (2.0, 4.0, False)),
}
EXIT_VARIANTS = {"SL2/TP2": (2.0, 2.0, False), "SL2/TP3": (2.0, 3.0, False), "SL2/TP4": (2.0, 4.0, False)}


def adx(df, n=14):
    h, l, c = df["High"], df["Low"], df["Close"]
    up, dn = h.diff(), -l.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    a = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean().values


def build(df):
    c = df["Close"]
    e20, e50, e200 = fa.ema(c, 20), fa.ema(c, 50), fa.ema(c, 200)
    r = fa.rsi(c).values
    macd = fa.ema(c, 12) - fa.ema(c, 26)
    hist = (macd - fa.ema(macd, 9)).values
    ev = {
        "ema50": b2.ev(np.where(c > e50, 1, -1)),
        "macd": b2.ev(np.where(hist > 0, 1, -1)),
        "rsi": b2.ev(np.where(r < 30, 1, np.where(r > 70, -1, 0))),
    }
    trend = np.where((e20 > e50) & (e50 > e200), 1, np.where((e20 < e50) & (e50 < e200), -1, 0))
    wk = c.resample("W-FRI").last().dropna()
    wsig = np.sign(wk - wk.ewm(span=10, adjust=False).mean()).shift(1)   # прошлая закрытая неделя
    wd = wsig.reindex(df.index, method="ffill").fillna(0).values
    f = dict(side200=np.sign((c - e200).values), trend=trend, adx=adx(df), rsi=r, week=wd)
    return ev, f, fa.atr(df).values


def filter_masks(rule, ev, f):
    s = ev[rule]
    n = len(s)
    conf = np.zeros(n, dtype=bool)
    for other, eo in ev.items():
        if other == rule:
            continue
        for k in range(3):
            sh = np.r_[np.zeros(k), eo[:n - k]]
            conf |= (sh == s) & (s != 0)
    return {
        "без фильтра": np.ones(n, dtype=bool),
        "по тренду EMA200": s == f["side200"],
        "по тренду EMA20/50/200": s == f["trend"],
        "по недельному тренду": s == f["week"],
        "сильный тренд (ADX>20)": f["adx"] > 20,
        "флэт (ADX<20)": f["adx"] < 20,
        "RSI не перегрет": ((s == 1) & (f["rsi"] < 65)) | ((s == -1) & (f["rsi"] > 35)),
        "подтверждено другим правилом": conf,
    }


def main():
    data = {}
    for name, tick in fa.PAIRS.items():
        try:
            df = fa.fetch(tick, "1d", period="5y")
            ev, f, atr = build(df)
            data[name] = (df, ev, f, atr)
            print(f"{name}: {len(df)} свечей")
        except Exception as e:
            print(f"{name}: ошибка — {e}")
    if not data:
        return
    recs = {}

    def collect(rule, mask_name, params):
        key = (rule, mask_name, params)
        if key in recs:
            return recs[key]
        res = []
        for nm, (df, ev, f, atr) in data.items():
            mid = (START + len(df)) // 2
            m = filter_masks(rule, ev, f)[mask_name]
            evs = np.where(m, ev[rule], 0)
            for i, r in b2.run(df, atr, evs, params, b2.SPREAD_PIPS[nm] * b2.PIP, START):
                res.append((nm, r, 0 if i < mid else 1))
        recs[key] = res
        return res

    def st(trades):
        if not trades:
            return 0, 0.0, 0.0, 0.0, 0.0
        r = np.array([t[1] for t in trades])
        a = np.array([t[1] for t in trades if t[2] == 0])
        b = np.array([t[1] for t in trades if t[2] == 1])
        return (len(r), float(np.mean(r > 0) * 100), float(r.mean()),
                float(a.mean()) if len(a) else 0.0, float(b.mean()) if len(b) else 0.0)

    good = []
    for rule, (title, params) in RULES.items():
        print(f"\n=== {title} (выход SL{params[0]:g}/TP{params[1]:g} ATR) ===")
        print(f"{'фильтр':32s}{'сделок':>7s}{'win%':>6s}{'ср.R':>7s}{'1-я':>7s}{'2-я':>7s}")
        base = st(collect(rule, "без фильтра", params))
        for fname in filter_masks(rule, data[next(iter(data))][1], data[next(iter(data))][2]):
            n, w, avg, a1, a2 = st(collect(rule, fname, params))
            better = (fname != "без фильтра" and n >= MIN_N and w > base[1] and avg > base[2]
                      and a1 > 0 and a2 > 0)
            if better:
                good.append((rule, title, fname, n, w, avg, base[1], base[2]))
            print(f"{fname:32s}{n:>7d}{w:>6.0f}{avg:>+7.2f}{a1:>+7.2f}{a2:>+7.2f}{'  <== лучше' if better else ''}")

    print("\n=== Обмен: процент прибыльных против размера прибыли (без фильтров) ===")
    print(f"{'правило':26s}" + "".join(f"{e:>22s}" for e in EXIT_VARIANTS))
    for rule, (title, _) in RULES.items():
        row = ""
        for ex, params in EXIT_VARIANTS.items():
            n, w, avg, _, _ = st(collect(rule, "без фильтра", params))
            row += f"{w:>8.0f}% {avg:>+6.2f}R ({n:>3d})"
        print(f"{title:26s}{row}")

    print("\n=== Вывод ===")
    if not good:
        print("Ни один фильтр не улучшил одновременно процент прибыльных и средний результат,")
        print("да ещё и в обеих половинах истории. Оставляем правила как есть.")
    else:
        for rule, title, fname, n, w, avg, bw, bavg in good:
            print(f"{title} + «{fname}»: win {bw:.0f}% -> {w:.0f}%, ср.R {bavg:+.2f} -> {avg:+.2f}, сделок {n}")
        print("Это кандидаты. Фильтров много, и часть улучшений может быть случайной,")
        print("поэтому добавляем только те, где улучшение заметное и держится в обеих половинах.")
    print("Прошлые результаты не гарантируют будущих.")


if __name__ == "__main__":
    main()
