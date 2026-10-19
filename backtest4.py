#!/usr/bin/env python3
"""
Бэктест №4: строгая проверка на 10 годах с контрольной выборкой.

Правила игры (заданы заранее, чтобы не подгонять результат):
  - Идей всего 6, у каждой ОДИН выход, выбранный до теста.
  - История делится на «разработку» (всё, кроме последних ~2 лет) и «контроль»
    (последние ~2 года). Контроль смотрим один раз, подбора по нему нет.
  - Идея считается кандидатом, только если: сделок >= 100, плюс на разработке
    (> +0.05R), плюс на контроле (> 0) и t-статистика >= 1.5.
  - Всё считается после спреда. «Случайный вход» показывает уровень шума.

Идеи: EMA50, MACD, RSI (текущие правила бота) + моментум 120 дней
(раз в месяц), Bollinger 20/2, Stochastic 20/80.
Кэрри (разницу ставок) не проверяем: бесплатной истории ставок нет.

Запуск:  python backtest4.py
"""
import numpy as np
import pandas as pd

import backtest2 as b2
import forex_analyzer as fa

START = 150
HOLDOUT = 520        # ~2 года дневных свечей
MIN_N = 100
ALT = (2.0, 2.0, False)   # выход с близким TP, только для справки о проценте прибыльных
IDEAS = [
    ("ema50", "Цена пересекла EMA50", (2.0, 4.0, False)),
    ("macd", "Пересечение MACD", (1.5, 3.0, False)),
    ("rsi", "Разворот RSI 30/70", (2.0, 4.0, False)),
    ("mom", "Моментум 120 дн. (раз в месяц)", (2.0, 4.0, False)),
    ("boll", "Bollinger 20/2 (возврат)", (2.0, 4.0, False)),
    ("stoch", "Stochastic 20/80", (2.0, 4.0, False)),
    ("rand", "Случайный вход (контроль)", (2.0, 4.0, False)),
]


def build(df):
    c, h, l = df["Close"], df["High"], df["Low"]
    n = len(df)
    e50 = fa.ema(c, 50)
    r = fa.rsi(c).values
    macd = fa.ema(c, 12) - fa.ema(c, 26)
    hist = (macd - fa.ema(macd, 9)).values
    ev = {
        "ema50": b2.ev(np.where(c > e50, 1, -1)),
        "macd": b2.ev(np.where(hist > 0, 1, -1)),
        "rsi": b2.ev(np.where(r < 30, 1, np.where(r > 70, -1, 0))),
    }
    months = df.index.to_period("M")
    first = np.r_[True, np.asarray(months[1:] != months[:-1])]
    ret = (c - c.shift(120)).values
    ev["mom"] = np.where(first & np.isfinite(ret), np.sign(np.nan_to_num(ret)), 0).astype(int)
    m, sd = c.rolling(20).mean(), c.rolling(20).std()
    ev["boll"] = b2.ev(np.where(c < m - 2 * sd, 1, np.where(c > m + 2 * sd, -1, 0)))
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    k = (100 * (c - ll) / (hh - ll).replace(0, np.nan)).rolling(3).mean().values
    ev["stoch"] = b2.ev(np.where(k < 20, 1, np.where(k > 80, -1, 0)))
    rng = np.random.default_rng(0)
    ev["rand"] = np.where(rng.random(n) < 0.05, rng.choice([-1, 1], n), 0)
    return ev, fa.atr(df).values


def summarize(rs):
    if len(rs) == 0:
        return 0, 0.0, 0.0, 0.0
    a = np.array(rs)
    t = a.mean() / (a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 and a.std() > 0 else 0.0
    return len(a), float(np.mean(a > 0) * 100), float(a.mean()), float(t)


def main():
    data = {}
    for name, tick in fa.PAIRS.items():
        try:
            df = fa.fetch(tick, "1d", period="10y")
            ev, atr = build(df)
            data[name] = (df, ev, atr)
            print(f"{name}: {len(df)} свечей ({df.index[0].date()} .. {df.index[-1].date()})")
        except Exception as e:
            print(f"{name}: ошибка — {e}")
    if not data:
        return

    def trades(key, params):
        out = []
        for nm, (df, ev, atr) in data.items():
            cut = len(df) - HOLDOUT
            for i, r in b2.run(df, atr, ev[key], params, b2.SPREAD_PIPS[nm] * b2.PIP, START):
                out.append((nm, r, i >= cut))
        return out

    print(f"\n=== Итоги по идеям (после спреда). Контроль = последние ~2 года ===")
    print(f"{'идея':32s}{'сделок':>7s}{'win%':>6s}{'ср.R':>7s}{'t':>6s}{'разраб.':>9s}{'контроль':>10s}{'TP2: win%/R':>14s}")
    passed = []
    for key, title, params in IDEAS:
        tr = trades(key, params)
        n, w, avg, t = summarize([x[1] for x in tr])
        dn, _, davg, _ = summarize([x[1] for x in tr if not x[2]])
        hn, _, havg, _ = summarize([x[1] for x in tr if x[2]])
        an, aw, aavg, _ = summarize([x[1] for x in trades(key, ALT)])
        ok = key != "rand" and n >= MIN_N and davg > 0.05 and havg > 0 and t >= 1.5
        if ok:
            passed.append((key, title, params))
        print(f"{title:32s}{n:>7d}{w:>6.0f}{avg:>+7.2f}{t:>6.1f}{davg:>+8.2f} {hn:>3d}{havg:>+6.2f}{aw:>9.0f}%/{aavg:+.2f}"
              .replace("  ", " ", 0) + ("  <== кандидат" if ok else ""))

    if passed:
        print("\n=== По парам (только кандидаты) ===")
        for key, title, params in passed:
            print(title)
            tr = trades(key, params)
            for nm in data:
                n, w, avg, _ = summarize([x[1] for x in tr if x[0] == nm])
                print(f"   {nm}: сделок {n:3d} | win {w:3.0f}% | {avg:+.2f}R")

    print("\n=== Вывод ===")
    if not passed:
        print("Ни одна идея не прошла все условия (плюс на разработке, плюс на контроле, t >= 1.5).")
        print("Значит, надёжного преимущества у простых дневных правил на этих парах нет;")
        print("бот остаётся информационным инструментом (уровни, SL/TP, новости, журнал).")
    else:
        print("Кандидаты:", ", ".join(t for _, t, _ in passed))
        print("Это повод для демо-торговли и наблюдения по журналу, а не гарантия.")
    print("Прошлые результаты не гарантируют будущих.")


if __name__ == "__main__":
    main()
