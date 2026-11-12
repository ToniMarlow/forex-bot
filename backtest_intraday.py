#!/usr/bin/env python3
"""
Бэктест интрадей-режима (1H, сделка до 4 часов) с контрольной выборкой.

Что проверяем и как (всё задано заранее, чтобы не подгонять результат):
  - 3 правила бота на 1H: EMA50, Stochastic 20/80, RSI 30/70 + контроль «случайный вход».
  - Фильтры добавляются цепочкой: без фильтров -> +сессия -> +тренд 4H -> +ADX 4H.
    Так видно, что именно даёт пользу, а что нет.
  - 2 варианта выхода (SL/TP в ATR 1H), максимум 4 часа в сделке.
  - Всё после спреда. Сделки одной пары не пересекаются.
  - Последние 25% времени = КОНТРОЛЬ: на нём ничего не подбирали.
  - Кандидат: сделок >= 100, плюс на разработке (> +0.05R), плюс на контроле (> 0), t >= 2.
    Проверок здесь ~50, поэтому один-два случайных «кандидата» возможны даже на шуме:
    верь только тем, кто плюсовой на контроле и на большинстве пар.

Данные: Yahoo отдаёт часовые свечи максимум за 730 дней. Это мало: выводы осторожные.

Запуск:  python backtest_intraday.py
"""
import numpy as np
import pandas as pd

import forex_analyzer as fa
import intraday as idy

MIN_N = 100
HOLDOUT_SHARE = 0.25
VARIANTS = [("без фильтров", set()),
            ("+сессия", {"session"}),
            ("+тренд 4H", {"session", "trend"}),
            ("+ADX 4H", {"session", "trend", "adx"})]
TITLE = {"ema50": "EMA50", "stoch": "Stochastic 20/80", "rsi": "RSI 30/70", "rand": "случайный вход (контроль)"}


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
            df = idy.fetch_1h(tick, "730d")
            ev, aux = idy.build(df)
            cut = df.index[int(len(df) * (1 - HOLDOUT_SHARE))]
            data[name] = (df, ev, aux, cut)
            print(f"{name}: {len(df)} часовых свечей ({df.index[0].date()} .. {df.index[-1].date()}), "
                  f"контроль с {cut.date()}")
        except Exception as e:
            print(f"{name}: ошибка — {e}")
    if not data:
        return

    def collect(rule, flt, exit_key):
        out = []
        for nm, (df, ev, aux, cut) in data.items():
            sp = idy.SPREAD_PIPS[nm] * idy.PIP
            for i, r, pts in idy.trades_for(df, aux, ev, rule, flt, exit_key, sp):
                out.append((nm, r, pts, df.index[i] >= cut))
        return out

    passed = []
    for exit_key, (sl_a, tp_a) in idy.EXITS.items():
        print(f"\n=== Выход {exit_key}: SL {sl_a} ATR / TP {tp_a} ATR, максимум {idy.HOLD_BARS} ч ===")
        print(f"{'правило':26s}{'фильтры':14s}{'сделок':>7s}{'win%':>6s}{'ср.R':>7s}{'t':>6s}"
              f"{'разраб.':>9s}{'контроль':>11s}")
        for rule in list(idy.RULES) + ["rand"]:
            for vname, flt in (VARIANTS if rule != "rand" else VARIANTS[:2]):
                tr = collect(rule, flt, exit_key)
                n, w, avg, t = summarize([x[1] for x in tr])
                dn, _, davg, _ = summarize([x[1] for x in tr if not x[3]])
                hn, _, havg, _ = summarize([x[1] for x in tr if x[3]])
                ok = rule != "rand" and n >= MIN_N and davg > 0.05 and havg > 0 and t >= 2.0
                if ok:
                    passed.append((rule, vname, flt, exit_key))
                print(f"{TITLE[rule]:26s}{vname:14s}{n:>7d}{w:>6.0f}{avg:>+7.2f}{t:>6.1f}"
                      f"{davg:>+8.2f} {hn:>3d}{havg:>+7.2f}" + ("  <== кандидат" if ok else ""))

    print("\n=== Детали: все 3 правила вместе, все фильтры, по классам подтверждений ===")
    full = {"session", "trend", "adx"}
    for exit_key in idy.EXITS:
        allt = []
        for rule in idy.RULES:
            allt += collect(rule, full, exit_key)
        print(f"\nВыход {exit_key}:")
        for lab, lo, hi in (("класс A (3-4 подтв.)", 3, 4), ("класс B (2 подтв.)", 2, 2), ("класс C (0-1)", 0, 1)):
            sel = [x for x in allt if lo <= x[2] <= hi]
            n, w, avg, t = summarize([x[1] for x in sel])
            hn, _, havg, _ = summarize([x[1] for x in sel if x[3]])
            print(f"  {lab:22s} сделок {n:4d} | win {w:3.0f}% | {avg:+.2f}R | контроль {hn:3d} сд. {havg:+.2f}R")
        print("  по парам (классы A+B):")
        for nm in data:
            sel = [x for x in allt if x[0] == nm and x[2] >= 2]
            n, w, avg, _ = summarize([x[1] for x in sel])
            print(f"    {nm}: сделок {n:3d} | win {w:3.0f}% | {avg:+.2f}R")

    print("\n=== Вывод ===")
    if not passed:
        print("Ни одна связка не прошла все условия (плюс на разработке, плюс на контроле, t >= 2).")
        print("Значит, надёжного преимущества на 1H за последние 2 года не найдено.")
        print("Интрадей-сигналы в боте остаются экспериментом: только демо-счёт, смотри журнал.")
    else:
        print("Кандидаты (правило / фильтры / выход):")
        for rule, vname, _, ek in passed:
            print(f"  - {TITLE[rule]} / {vname} / выход {ek}")
        print("Это повод для демо-торговли и наблюдения за журналом, а не гарантия прибыли.")
    print("Прошлые результаты не гарантируют будущих. Проверок много: опирайся на контроль и на стабильность по парам.")


if __name__ == "__main__":
    main()
