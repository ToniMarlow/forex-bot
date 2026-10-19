#!/usr/bin/env python3
"""
Ежедневный анализ рынка: GBP/USD, EUR/USD, GBP/AUD, EUR/NZD, EUR/CAD.
Свечи + графические паттерны + индикаторы + фундаментал (ставки) + новости/сентимент.

Установка:   pip install yfinance pandas numpy
Запуск:      python forex_analyzer.py --tf 4h
             python forex_analyzer.py --tf 1d
Результат:   таблица в консоли + файл report_ДАТА_таймфрейм.html
Telegram:    задай переменные окружения TELEGRAM_BOT_TOKEN и TELEGRAM_CHAT_ID,
             сигналы BUY/SELL придут в Telegram автоматически.

Это инструмент поддержки решений, а не гарантия прибыли.
"""
import argparse
import datetime as dt
import html
import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import yfinance as yf

# ------------------------- НАСТРОЙКИ -------------------------
PAIRS = {
    "GBP/USD": "GBPUSD=X",
    "EUR/USD": "EURUSD=X",
    "GBP/AUD": "GBPAUD=X",
    "EUR/NZD": "EURNZD=X",
    "EUR/CAD": "EURCAD=X",
}

# Ключевые ставки центробанков, %. ОБНОВЛЯЙ ВРУЧНУЮ раз в месяц
# (значения ниже примерные, проверь на сайтах центробанков).
POLICY_RATES = {"USD": 4.0, "EUR": 2.0, "GBP": 4.0, "AUD": 3.6, "NZD": 2.5, "CAD": 2.5}

NEWS_QUERY = {
    "USD": "US dollar Federal Reserve",
    "EUR": "euro ECB",
    "GBP": "pound sterling Bank of England",
    "AUD": "Australian dollar RBA",
    "NZD": "New Zealand dollar RBNZ",
    "CAD": "Canadian dollar Bank of Canada",
}

# Веса компонентов (в сумме 1.0)
WEIGHTS = {"technical": 0.35, "candles": 0.20, "patterns": 0.15,
           "fundamental": 0.15, "sentiment": 0.15}
SIGNAL_THRESHOLD = 25  # |итог| >= 25 -> BUY/SELL, иначе WAIT

POS_WORDS = ["hawkish", "rate hike", "hikes", "strong", "rises", "rallies", "surge",
             "gains", "beats", "robust", "jumps", "climbs", "higher inflation",
             "sticky inflation", "upbeat", "optimism", "boost"]
NEG_WORDS = ["dovish", "rate cut", "cuts rates", "weak", "falls", "slumps", "drops",
             "recession", "misses", "slows", "loses", "declines", "tumbles",
             "slides", "downbeat", "pessimism", "unemployment rises", "slowdown"]


# ------------------------- ДАННЫЕ -------------------------
def fetch(ticker, tf):
    if tf == "4h":
        df = yf.download(ticker, period="60d", interval="1h", progress=False, auto_adjust=True)
    else:
        df = yf.download(ticker, period="2y", interval="1d", progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError("нет данных")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    if tf == "4h":
        df = df.resample("4h").agg({"Open": "first", "High": "max",
                                    "Low": "min", "Close": "last"}).dropna()
    return df


# ------------------------- ИНДИКАТОРЫ -------------------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n=14):
    pc = df["Close"].shift()
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(),
                    (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def technical_score(df):
    c = df["Close"]
    e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200)
    r = rsi(c).iloc[-1]
    macd = ema(c, 12) - ema(c, 26)
    hist = macd - ema(macd, 9)
    score, notes = 0.0, []

    if e20.iloc[-1] > e50.iloc[-1] > e200.iloc[-1]:
        score += 40; notes.append("восходящий тренд (EMA20>50>200)")
    elif e20.iloc[-1] < e50.iloc[-1] < e200.iloc[-1]:
        score -= 40; notes.append("нисходящий тренд (EMA20<50<200)")
    else:
        notes.append("тренд смешанный")

    score += 15 if c.iloc[-1] > e50.iloc[-1] else -15
    score += 20 if hist.iloc[-1] > 0 else -20
    score += 10 if hist.iloc[-1] > hist.iloc[-2] else -10
    notes.append(f"MACD {'выше' if hist.iloc[-1] > 0 else 'ниже'} сигнала")

    if r < 30:
        score += 15; notes.append(f"RSI {r:.0f} (перепродан)")
    elif r > 70:
        score -= 15; notes.append(f"RSI {r:.0f} (перекуплен)")
    else:
        notes.append(f"RSI {r:.0f}")
    return float(np.clip(score, -100, 100)), notes


# ------------------------- СВЕЧНЫЕ ПАТТЕРНЫ -------------------------
def candle_score(df):
    o, h, l, c = (df[k].values for k in ["Open", "High", "Low", "Close"])
    i = -1
    body = lambda k: abs(c[k] - o[k])
    rng = lambda k: max(h[k] - l[k], 1e-12)
    bull = lambda k: c[k] > o[k]
    bear = lambda k: c[k] < o[k]
    down = c[-6] > c[-2]   # перед последней свечой был спад
    up = c[-6] < c[-2]
    score, found = 0.0, []

    if bear(-2) and bull(i) and o[i] <= c[-2] and c[i] >= o[-2]:
        score += 2; found.append("бычье поглощение")
    if bull(-2) and bear(i) and o[i] >= c[-2] and c[i] <= o[-2]:
        score -= 2; found.append("медвежье поглощение")

    lw = min(o[i], c[i]) - l[i]
    uw = h[i] - max(o[i], c[i])
    if down and lw >= 2 * body(i) and uw <= 0.5 * body(i) + 1e-12 and body(i) > 0:
        score += 1.5; found.append("молот")
    if up and uw >= 2 * body(i) and lw <= 0.5 * body(i) + 1e-12 and body(i) > 0:
        score -= 1.5; found.append("падающая звезда")

    if body(i) <= 0.1 * rng(i):
        found.append("доджи (неопределённость)")

    mid1 = (o[-3] + c[-3]) / 2
    if bear(-3) and body(-2) < 0.4 * body(-3) and bull(i) and c[i] > mid1:
        score += 2.5; found.append("утренняя звезда")
    if bull(-3) and body(-2) < 0.4 * body(-3) and bear(i) and c[i] < mid1:
        score -= 2.5; found.append("вечерняя звезда")

    return float(np.clip(score, -3, 3) / 3 * 100), found or ["явных свечных моделей нет"]


# ------------------------- ГРАФИЧЕСКИЕ ПАТТЕРНЫ / УРОВНИ -------------------------
def pivots(s, n=5, high=True):
    out = []
    v = s.values
    for k in range(n, len(v) - n):
        w = v[k - n:k + n + 1]
        if (high and v[k] == w.max()) or ((not high) and v[k] == w.min()):
            out.append((k, v[k]))
    return out


def pattern_score(df):
    c = df["Close"].iloc[-1]
    a = atr(df).iloc[-1]
    ph, pl = pivots(df["High"], 5, True), pivots(df["Low"], 5, False)
    score, found = 0.0, []

    if len(ph) >= 2:
        (i1, h1), (i2, h2) = ph[-2], ph[-1]
        trough = df["Low"].values[i1:i2 + 1].min()
        if abs(h1 - h2) / h1 < 0.003 and c < trough:
            score -= 60; found.append("двойная вершина (пробой вниз)")
    if len(pl) >= 2:
        (i1, l1), (i2, l2) = pl[-2], pl[-1]
        peak = df["High"].values[i1:i2 + 1].max()
        if abs(l1 - l2) / l1 < 0.003 and c > peak:
            score += 60; found.append("двойное дно (пробой вверх)")

    if ph:
        res = ph[-1][1]
        if c > res:
            score += 30; found.append("пробой последнего максимума")
        elif 0 <= res - c < 0.5 * a:
            score -= 20; found.append("цена у сопротивления")
    if pl:
        sup = pl[-1][1]
        if c < sup:
            score -= 30; found.append("пробой последнего минимума")
        elif 0 <= c - sup < 0.5 * a:
            score += 20; found.append("цена у поддержки")
    return float(np.clip(score, -100, 100)), found or ["явных паттернов нет"]


# ------------------------- ФУНДАМЕНТАЛ -------------------------
def fundamental_score(base, quote):
    diff = POLICY_RATES[base] - POLICY_RATES[quote]
    return float(np.clip(diff * 25, -100, 100)), [
        f"ставки: {base} {POLICY_RATES[base]}% vs {quote} {POLICY_RATES[quote]}% (разница {diff:+.2f})"]


# ------------------------- НОВОСТИ / СЕНТИМЕНТ -------------------------
_news_cache = {}


def currency_news(cur):
    if cur in _news_cache:
        return _news_cache[cur]
    q = urllib.parse.quote(NEWS_QUERY[cur] + " when:2d")
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                                    timeout=15) as r:
            root = ET.fromstring(r.read())
        titles = [(i.findtext("title") or "") for i in root.iter("item")][:15]
    except Exception:
        titles = []
    _news_cache[cur] = titles
    return titles


def headline_score(t):
    t = t.lower()
    return sum(w in t for w in POS_WORDS) - sum(w in t for w in NEG_WORDS)


def currency_sentiment(cur):
    titles = currency_news(cur)
    if not titles:
        return 0.0, titles
    return float(np.clip(np.mean([headline_score(t) for t in titles]) * 50, -100, 100)), titles


def sentiment_score(base, quote):
    sb, tb = currency_sentiment(base)
    sq, tq = currency_sentiment(quote)
    notes = [f"новости: {base} {sb:+.0f}, {quote} {sq:+.0f} ({len(tb) + len(tq)} заголовков)"]
    heads = (tb[:2] + tq[:2])
    return float(np.clip(sb - sq, -100, 100)), notes, heads


# ------------------------- ИТОГ -------------------------
def analyze(name, ticker, tf):
    base, quote = name.split("/")
    df = fetch(ticker, tf)
    t, tn = technical_score(df)
    cd, cn = candle_score(df)
    pt, pn = pattern_score(df)
    fu, fn = fundamental_score(base, quote)
    se, sn, heads = sentiment_score(base, quote)
    parts = {"technical": t, "candles": cd, "patterns": pt, "fundamental": fu, "sentiment": se}
    total = sum(parts[k] * WEIGHTS[k] for k in WEIGHTS)

    signal = "BUY" if total >= SIGNAL_THRESHOLD else "SELL" if total <= -SIGNAL_THRESHOLD else "WAIT"
    sign = 1 if total > 0 else -1
    agree = sum(1 for v in parts.values() if v * sign > 10)
    confidence = int(min(100, abs(total) * 1.6 + agree * 6))

    price, a = df["Close"].iloc[-1], atr(df).iloc[-1]
    sl = tp = None
    if signal == "BUY":
        sl, tp = price - 1.5 * a, price + 3 * a
    elif signal == "SELL":
        sl, tp = price + 1.5 * a, price - 3 * a

    return dict(pair=name, price=price, signal=signal, total=total, confidence=confidence,
                agree=agree, parts=parts, sl=sl, tp=tp,
                notes=tn + cn + pn + fn + sn, heads=heads)


def render_html(results, tf):
    colors = {"BUY": "#1a9e55", "SELL": "#d93b3b", "WAIT": "#888"}
    rows = ""
    for r in results:
        lv = f"SL {r['sl']:.5f} / TP {r['tp']:.5f}" if r["sl"] else "—"
        rows += (f"<div class='c'><h2>{r['pair']} <span style='color:{colors[r['signal']]}'>"
                 f"{r['signal']}</span></h2><p>Цена {r['price']:.5f} · уверенность {r['confidence']}% · "
                 f"итог {r['total']:+.0f}</p><p>{lv}</p>"
                 f"<p>{' · '.join(f'{k} {v:+.0f}' for k, v in r['parts'].items())}</p>"
                 f"<ul>{''.join(f'<li>{html.escape(n)}</li>' for n in r['notes'])}</ul>"
                 f"<small>{'<br>'.join(html.escape(h) for h in r['heads'])}</small></div>")
    return (f"<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
            f"<style>body{{font-family:sans-serif;max-width:700px;margin:auto;padding:12px}}"
            f".c{{border:1px solid #ccc;border-radius:10px;padding:10px;margin:10px 0}}</style>"
            f"<h1>Анализ рынка {dt.date.today()} ({tf})</h1>{rows}"
            f"<p><i>Не является финансовой рекомендацией. Всегда используй стоп-лосс.</i></p>")


# ------------------------- TELEGRAM -------------------------
def send_telegram(text):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("Telegram: не заданы TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID, отправка пропущена")
        return
    data = json.dumps({"chat_id": chat_id, "text": text[:4000]}).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        urllib.request.urlopen(req, timeout=20).read()
        print("Telegram: отправлено")
    except Exception as e:
        print(f"Telegram: ошибка отправки — {e}")


def telegram_text(results, tf):
    icons = {"BUY": "🟢", "SELL": "🔴", "WAIT": "⚪"}
    active = [r for r in results if r["signal"] != "WAIT"]
    head = f"📊 Анализ рынка {dt.date.today()} ({tf})\n"
    if not active:
        waits = ", ".join(r["pair"] for r in results)
        return head + f"\nСегодня чётких сигналов нет, ждём: {waits}"
    blocks = []
    for r in active:
        b = (f"{icons[r['signal']]} {r['signal']} {r['pair']}\n"
             f"Цена: {r['price']:.5f}\nУверенность: {r['confidence']}%\n"
             f"SL: {r['sl']:.5f}  TP: {r['tp']:.5f}\n"
             f"Причины: " + "; ".join(r["notes"][:5]))
        blocks.append(b)
    waits = [r["pair"] for r in results if r["signal"] == "WAIT"]
    tail = ("\n\nWAIT: " + ", ".join(waits)) if waits else ""
    return head + "\n" + "\n\n".join(blocks) + tail + "\n\nНе финансовая рекомендация. Используй стоп-лосс."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", choices=["4h", "1d"], default="1d")
    tf = ap.parse_args().tf

    results = []
    for name, tick in PAIRS.items():
        try:
            results.append(analyze(name, tick, tf))
        except Exception as e:
            print(f"{name}: ошибка — {e}")

    print(f"\n=== Анализ {dt.date.today()} | таймфрейм {tf} ===")
    for r in results:
        print(f"\n{r['pair']}  {r['signal']}  (уверенность {r['confidence']}%, итог {r['total']:+.0f})")
        print("  " + " | ".join(f"{k} {v:+.0f}" for k, v in r["parts"].items()))
        if r["sl"]:
            print(f"  вход ~{r['price']:.5f}  SL {r['sl']:.5f}  TP {r['tp']:.5f}")
        for n in r["notes"]:
            print("   -", n)

    fn = f"report_{dt.date.today()}_{tf}.html"
    with open(fn, "w", encoding="utf-8") as f:
        f.write(render_html(results, tf))
    print(f"\nОтчёт сохранён: {fn}")
    if results:
        send_telegram(telegram_text(results, tf))
    print("Не финансовая рекомендация. Всегда ставь стоп-лосс.")


if __name__ == "__main__":
    main()
