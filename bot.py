#!/usr/bin/env python3
"""
Telegram-бот по форексу: работает сам, 24/7 (нужен сервер, см. инструкцию).

Что умеет:
  - любой человек пишет боту /start и сразу зарегистрирован;
  - кнопки: выбрать пару и мгновенно получить данные по ней (цена, правила,
    уровни SL/TP, новости);
  - у каждого свой набор пар для ежедневной рассылки (кнопка «Мои пары»);
  - ежедневный разбор в заданное время (по умолчанию 02:00 UTC = 07:00 Ташкент);
  - журнал сигналов: бот сам проверяет, чем закончились прошлые сигналы,
    и раз в неделю (понедельник) присылает итоги.

Правила сигналов (бэктест на 10 годах дневок, см. backtest4.py):
  1) цена пересекла EMA50        (SL 2 ATR / TP 4 ATR) - основное
  2) Stochastic вошёл в зону 20/80 (SL 2 ATR / TP 4 ATR) - основное
  3) разворот по RSI 30/70       (SL 2 ATR / TP 4 ATR) - слабее

Переменные окружения:
  TELEGRAM_BOT_TOKEN  токен бота (обязательно)
  ALLOWED_USERS       необязательно: id через запятую, если бот должен быть закрытым
  DAILY_HOUR_UTC      час ежедневной рассылки по UTC (по умолчанию 2)
  ADMIN_IDS           id администраторов через запятую (команда /status); по умолчанию = ALLOWED_USERS
  ALERT_EVERY_SECONDS как часто проверять цену для оповещений (по умолчанию 300)
  DB_PATH             путь к базе (по умолчанию /data/bot.db, если есть папка /data)
"""
import datetime as dt
import html
import json
import os
import sqlite3
import threading
import time
import traceback
import urllib.request

import numpy as np
import pandas as pd

import forex_analyzer as fa

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
DB_PATH = os.environ.get("DB_PATH") or ("/data/bot.db" if os.path.isdir("/data") else "bot.db")
ALLOWED = {int(x) for x in os.environ.get("ALLOWED_USERS", "").replace(" ", "").split(",") if x}
DAILY_HOUR_UTC = int(os.environ.get("DAILY_HOUR_UTC", "2"))
ADMIN = {int(x) for x in os.environ.get("ADMIN_IDS", "").replace(" ", "").split(",") if x} or ALLOWED
ALERT_EVERY = int(os.environ.get("ALERT_EVERY_SECONDS", "300"))   # как часто проверять цену для оповещений

PIP = 0.0001
SPREAD_PIPS = {"GBP/USD": 1.2, "EUR/USD": 0.9, "GBP/AUD": 2.8, "EUR/NZD": 3.5, "EUR/CAD": 2.5}
MAXHOLD = 20
CACHE_SECONDS = 600

RULES = {
    "ema50": dict(title="Цена пересекла EMA50", sl=2.0, tp=4.0,
                  bt="Бэктест 10 лет: ~50% прибыльных, около +0.1R на сделку"),
    "stoch": dict(title="Stochastic 20/80", sl=2.0, tp=4.0,
                  bt="Бэктест 10 лет: ~49% прибыльных, около +0.1R на сделку"),
    "rsi": dict(title="Разворот по RSI (30/70)", sl=2.0, tp=4.0,
                bt="Бэктест 10 лет: ~47% прибыльных, около +0.06R (слабее)"),
}
ICON = {1: "🟢 BUY", -1: "🔴 SELL"}
SHORT = {"ema50": "EMA50", "stoch": "Stochastic", "rsi": "RSI"}
AGE = {0: "на последней закрытой свече", 1: "1 день назад", 2: "2 дня назад"}
KEY = {p.replace("/", ""): p for p in fa.PAIRS}

db = None
lock = threading.Lock()
_cache = {}


def log(*a):
    print(dt.datetime.now().strftime("%H:%M:%S"), *a, flush=True)


# ------------------------- БАЗА -------------------------
def init_db():
    global db
    db = sqlite3.connect(DB_PATH, check_same_thread=False)
    db.executescript("""
    CREATE TABLE IF NOT EXISTS users(chat_id INTEGER PRIMARY KEY, name TEXT, joined TEXT, daily INTEGER DEFAULT 1);
    CREATE TABLE IF NOT EXISTS subs(chat_id INTEGER, pair TEXT, PRIMARY KEY(chat_id, pair));
    CREATE TABLE IF NOT EXISTS signals(id INTEGER PRIMARY KEY AUTOINCREMENT, pair TEXT, rule TEXT, dir INTEGER,
        bar_date TEXT, entry REAL, sl REAL, tp REAL, risk REAL, spread REAL,
        status TEXT DEFAULT 'open', r REAL, UNIQUE(pair, rule, bar_date));
    CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
    CREATE TABLE IF NOT EXISTS alerts_sent(chat_id INTEGER, pair TEXT, key TEXT, day TEXT,
        PRIMARY KEY(chat_id, pair, key, day));
    """)
    for col, ddl in (("kb", "INTEGER DEFAULT 0"), ("deposit", "REAL"),
                     ("risk", "REAL DEFAULT 1"), ("alerts", "INTEGER DEFAULT 0")):
        try:
            db.execute(f"ALTER TABLE users ADD COLUMN {col} {ddl}")
        except sqlite3.OperationalError:
            pass
    db.commit()


def q(sql, args=()):
    with lock:
        cur = db.execute(sql, args)
        rows = cur.fetchall()
        db.commit()
        return rows


def get_meta(k):
    r = q("SELECT v FROM meta WHERE k=?", (k,))
    return r[0][0] if r else None


def set_meta(k, v):
    q("INSERT OR REPLACE INTO meta VALUES(?,?)", (k, v))


def ensure_user(chat_id, name):
    if q("SELECT 1 FROM users WHERE chat_id=?", (chat_id,)):
        return False
    q("INSERT INTO users(chat_id, name, joined) VALUES(?,?,?)", (chat_id, name, dt.date.today().isoformat()))
    for p in fa.PAIRS:
        q("INSERT OR IGNORE INTO subs VALUES(?,?)", (chat_id, p))
    log("новый пользователь", chat_id, name)
    return True


# ------------------------- TELEGRAM API -------------------------
def api(method, **params):
    params = {k: v for k, v in params.items() if v is not None}
    req = urllib.request.Request(f"https://api.telegram.org/bot{TOKEN}/{method}",
                                 data=json.dumps(params).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=45) as r:
            return json.loads(r.read())
    except Exception as e:
        if "not modified" not in str(e):
            log(f"API {method}: {e}")
        return None


def clip(text, limit=3900):
    """Режем по строкам, чтобы не оборвать HTML-тег посередине."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit("\n", 1)[0] + "\n…"


def send(chat_id, text, kb=None):
    return api("sendMessage", chat_id=chat_id, text=clip(text), reply_markup=kb, parse_mode="HTML")


def edit(chat_id, msg_id, text, kb=None):
    return api("editMessageText", chat_id=chat_id, message_id=msg_id, text=clip(text), reply_markup=kb,
               parse_mode="HTML")


# ------------------------- АНАЛИЗ -------------------------
def fmt(x):
    return f"{x:.5f}"


def load(pair):
    last = None
    for attempt in range(3):
        try:
            full = fa.fetch(fa.PAIRS[pair], "1d", period="2y")
            break
        except Exception as e:
            last = e
            time.sleep(2 * (attempt + 1))
    else:
        raise last
    today = dt.datetime.now(dt.timezone.utc).date()
    df = full.iloc[:-1] if full.index[-1].date() >= today else full
    return full, df


def _solve(f, lo, hi, target):
    """f возрастает; ищем x, где f(x) впервые достигает target."""
    for _ in range(60):
        mid = (lo + hi) / 2
        if f(mid) >= target:
            hi = mid
        else:
            lo = mid
    return hi


def calc_triggers(df, rv_last, k_last, atr):
    """Цены, за которыми ДНЕВНАЯ свеча должна закрыться, чтобы сработало правило."""
    c = df["Close"]
    close = float(c.iloc[-1])
    out = []

    # 1) EMA50: знак (close - EMA50) меняется ровно на уровне прошлой EMA50
    e50 = float(fa.ema(c, 50).iloc[-1])
    sd = -1 if close > e50 else 1
    out.append(dict(rule="ema50", dir=sd, side=sd, level=e50))

    # 2) Stochastic: считаем %K завтрашней свечи, считая её диапазон минимальным
    #    (от вчерашнего закрытия до цены закрытия), поэтому уровень приблизительный
    hi, lo = df["High"], df["Low"]
    ll14, hh14 = lo.rolling(14).min(), hi.rolling(14).max()
    raw = (c - ll14) / (hh14 - ll14).replace(0, np.nan) * 100
    r1, r2 = float(raw.iloc[-1]), float(raw.iloc[-2])
    hh13, ll13 = float(hi.iloc[-13:].max()), float(lo.iloc[-13:].min())

    def k_at(x):
        hh, ll = max(hh13, x, close), min(ll13, x, close)
        rawx = 100 * (x - ll) / (hh - ll) if hh > ll else 50.0
        return (r1 + r2 + rawx) / 3

    lo_x, hi_x = close - 10 * atr, close + 10 * atr
    if np.isfinite(r1) and np.isfinite(r2) and np.isfinite(k_last):
        if k_last >= 20 and k_at(lo_x) <= 20 < k_at(hi_x):
            out.append(dict(rule="stoch", dir=1, side=-1, level=_solve(k_at, lo_x, hi_x, 20)))
        if k_last <= 80 and k_at(lo_x) < 80 <= k_at(hi_x):
            out.append(dict(rule="stoch", dir=-1, side=1, level=_solve(k_at, lo_x, hi_x, 80)))

    # 3) RSI: цена, при которой RSI на закрытии пересечёт 70 (SELL) или 30 (BUY)
    d = c.diff()
    up = float(d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
    dn = float((-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])

    def rsi_at(x):
        dd = x - close
        u, v = up * 13 / 14 + max(dd, 0) / 14, dn * 13 / 14 + max(-dd, 0) / 14
        return 100.0 if v <= 0 else 100 - 100 / (1 + u / v)

    if rv_last <= 70 and rsi_at(close + 10 * atr) >= 70:
        out.append(dict(rule="rsi", dir=-1, side=1, level=_solve(rsi_at, close, close + 10 * atr, 70)))
    if rv_last >= 30 and rsi_at(close - 10 * atr) <= 30:
        out.append(dict(rule="rsi", dir=1, side=-1, level=_solve(rsi_at, close - 10 * atr, close, 30)))
    return out


def analyze_pair(pair, force=False):
    now = time.time()
    if not force and pair in _cache and now - _cache[pair][0] < CACHE_SECONDS:
        return _cache[pair][1]
    full, df = load(pair)
    c = df["Close"]
    e50 = fa.ema(c, 50)
    rv = fa.rsi(c).values
    ll14, hh14 = df["Low"].rolling(14).min(), df["High"].rolling(14).max()
    kv = ((c - ll14) / (hh14 - ll14).replace(0, np.nan) * 100).rolling(3).mean().values
    states = {
        "ema50": np.sign((c - e50).values),
        "stoch": np.where(kv < 20, 1, np.where(kv > 80, -1, 0)),
        "rsi": np.where(rv < 30, 1, np.where(rv > 70, -1, 0)),
    }
    n, events = len(df), []
    for rule, st in states.items():
        for age in range(3):
            i = n - 1 - age
            if i < 1:
                break
            if st[i] != 0 and st[i] != st[i - 1]:
                events.append(dict(rule=rule, dir=int(st[i]), age=age,
                                   bar=df.index[i].strftime("%Y-%m-%d"), entry=float(c.iloc[i])))
                break
    base, quote = pair.split("/")
    price, atr_v = float(full["Close"].iloc[-1]), float(fa.atr(df).iloc[-1])
    trig = []
    for t in calc_triggers(df, float(rv[-1]), float(kv[-1]), atr_v):
        if abs(t["level"] - price) <= 3 * atr_v:
            t["past"] = (price > t["level"]) if t["side"] == 1 else (price < t["level"])
            trig.append(t)
    trig.sort(key=lambda t: abs(t["level"] - price))
    res = dict(pair=pair, price=price, atr=atr_v, bar=df.index[-1].strftime("%Y-%m-%d"),
               events=events, rsi=float(rv[-1]), ema_side=int(states["ema50"][-1]),
               stoch=float(kv[-1]), trend=fa.trend_dir(df), triggers=trig,
               asof=dt.datetime.now(dt.timezone.utc).strftime("%d.%m %H:%M UTC"),
               news=fa.upcoming_events({base, quote}))
    _cache[pair] = (now, res)
    return res


def levels(price, atr, d, sl_a, tp_a):
    return price - d * sl_a * atr, price + d * tp_a * atr


EVENT_WHY = {
    ("ema50", 1): "цена закрылась выше EMA50, разворот вверх",
    ("ema50", -1): "цена закрылась ниже EMA50, разворот вниз",
    ("stoch", 1): "Stochastic вошёл в зону перепроданности (ниже 20), ждём отскок вверх",
    ("stoch", -1): "Stochastic вошёл в зону перекупленности (выше 80), ждём откат вниз",
    ("rsi", 1): "RSI упал ниже 30 (перепроданность), ждём отскок вверх",
    ("rsi", -1): "RSI поднялся выше 70 (перекупленность), ждём откат вниз",
}
WAIT_WHY = {
    ("ema50", 1): "EMA50: цена вернётся выше средней, разворот вверх",
    ("ema50", -1): "EMA50: цена уйдёт ниже средней, разворот вниз",
    ("stoch", 1): "Stochastic: уйдёт в зону перепроданности (ниже 20), ждём отскок",
    ("stoch", -1): "Stochastic: уйдёт в зону перекупленности (выше 80), ждём откат",
    ("rsi", 1): "RSI: упадёт ниже 30 (перепроданность), ждём отскок",
    ("rsi", -1): "RSI: поднимется выше 70 (перекупленность), ждём откат",
}


def code(x):
    return f"<code>{fmt(x)}</code>"


def trig_lines(res, limit=3, rk=None):
    out, p, a = [], res["price"], res["atr"]
    for t in res["triggers"][:limit]:
        d, rule = t["dir"], RULES[t["rule"]]
        entry = p if t["past"] else t["level"]
        sl, tp = levels(entry, a, d, rule["sl"], rule["tp"])
        where = "выше" if t["side"] == 1 else "ниже"
        action = "Покупка" if d == 1 else "Продажа"
        note = "цена уже за уровнем, ждём закрытие дня" if t["past"] else \
            f"{abs(t['level'] - p) / PIP:.0f} п. от цены"
        approx = " (уровень приблизительный)" if t["rule"] == "stoch" else ""
        out += [f"{'🟢' if d == 1 else '🔴'} <b>{action}</b>, если свеча закроется <b>{where}</b> {code(t['level'])}",
                f"   {WAIT_WHY[(t['rule'], d)]}{approx}",
                f"   {note}",
                f"   SL {code(sl)} · TP {code(tp)}{lot_suffix(res['pair'], entry, sl, rk)}", ""]
    return out[:-1] if out else out


def pair_text(res, with_triggers=True, rk=None):
    p, price, a = res["pair"], res["price"], res["atr"]
    tr = {1: "вверх", -1: "вниз", 0: "смешанный"}[res["trend"]]
    lines = [f"📊 <b>{p}</b> · {code(price)}  <i>({res['asof']})</i>",
             f"Тренд {tr} · цена {'выше' if res['ema_side'] > 0 else 'ниже'} EMA50 · "
             f"RSI {res['rsi']:.0f} · Stoch {res['stoch']:.0f}", ""]
    if res["events"]:
        lines.append("✅ <b>Есть сигнал</b>")
        for e in res["events"]:
            rule = RULES[e["rule"]]
            sl, tp = levels(price, a, e["dir"], rule["sl"], rule["tp"])
            when = "" if e["age"] == 0 else f" ({AGE[e['age']]})"
            lines += [f"{ICON[e['dir']].replace(' ', ' <b>')}</b> · {EVENT_WHY[(e['rule'], e['dir'])]}{when}",
                      f"Вход {code(price)} · SL {code(sl)} · TP {code(tp)}{lot_suffix(p, price, sl, rk)}"]
        lines.append("Риск : цель = 1 : 2. Если сделка пойдёт против тебя, потеря около 1R, цель около 2R.")
    else:
        lines.append("⚪ Сигнала сейчас нет.")
    if with_triggers and res["triggers"]:
        lines += ["", "🎯 <b>Что ждать</b> (смотрим, где закроется дневная свеча):", ""] + trig_lines(res, rk=rk)
    bs, bt = levels(price, a, 1, 2, 4)
    ss, st = levels(price, a, -1, 2, 4)
    lines += ["", "<b>Свой вход</b> (SL 2 ATR · TP 4 ATR):",
              f"🟢 Покупка: SL {code(bs)} · TP {code(bt)}{lot_suffix(p, price, bs, rk)}",
              f"🔴 Продажа: SL {code(ss)} · TP {code(st)}{lot_suffix(p, price, ss, rk)}"]
    if res["news"]:
        lines += ["", "⚠️ <b>Важные новости:</b> " + html.escape("; ".join(
            f"{c} {t} через {h:.0f}ч" for h, c, t in res["news"][:3]))]
        if res["news"][0][0] <= fa.CALENDAR_BLOCK_HOURS:
            lines.append("⛔ Новость совсем скоро: лучше подождать, спред и скачки цены вырастут.")
    if rk is None:
        lines += ["", "🧮 Задай депозит (/risk), и я буду писать объём лота под твой риск."]
    lines += ["", "<i>Не финансовая рекомендация. Всегда ставь стоп-лосс.</i>"]
    return "\n".join(lines)


def one_line(res):
    s = f"⚪ <b>{res['pair']}</b> · {code(res['price'])} · RSI {res['rsi']:.0f} — сигнала нет"
    if res["triggers"]:
        t = res["triggers"][0]
        s += (f"\n   ближайший: {'покупка' if t['dir'] == 1 else 'продажа'}, если свеча закроется "
              f"{'выше' if t['side'] == 1 else 'ниже'} {code(t['level'])} ({SHORT[t['rule']]})")
    return s


# ------------------------- РИСК И ОБЪЁМ ЛОТА -------------------------
CONV = {"AUD": ("AUDUSD=X", False), "NZD": ("NZDUSD=X", False), "CAD": ("USDCAD=X", True)}
_conv = {}


def usd_per_pip(pair):
    """Стоимость 1 пункта для 1 стандартного лота (100 000), в долларах. Приблизительно."""
    spec = CONV.get(pair.split("/")[1])
    if spec is None:
        return 10.0
    tick, inv = spec
    hit = _conv.get(tick)
    if hit and time.time() - hit[0] < 1800:
        rate = hit[1]
    else:
        rate = float(fa.fetch(tick, "1d", period="10d")["Close"].iloc[-1])
        _conv[tick] = (time.time(), rate)
    return 10.0 / rate if inv else 10.0 * rate


def lots(pair, entry, sl, rk):
    if not rk or not rk.get("deposit"):
        return None
    pips = abs(entry - sl) / PIP
    if pips <= 0:
        return None
    try:
        return rk["deposit"] * rk["risk"] / 100 / (pips * usd_per_pip(pair))
    except Exception:
        return None


def lot_suffix(pair, entry, sl, rk):
    v = lots(pair, entry, sl, rk)
    if v is None:
        return ""
    return " · объём ≈ " + (f"{v:.2f} лота" if v >= 0.01 else "<0.01 лота (депозит мал для такого стопа)")


def get_risk(chat_id):
    r = q("SELECT deposit, risk FROM users WHERE chat_id=?", (chat_id,))
    if not r or not r[0][0]:
        return None
    return {"deposit": float(r[0][0]), "risk": float(r[0][1] or 1)}


def parse_num(text, lo, hi):
    try:
        v = float(text.replace(",", ".").replace(" ", "").replace("$", ""))
    except ValueError:
        return None
    return v if lo <= v <= hi else None


def risk_text(chat_id):
    rk = get_risk(chat_id)
    if rk:
        cur = (f"Депозит: <b>${rk['deposit']:,.0f}</b> · риск на сделку: <b>{rk['risk']:g}%</b> "
               f"(≈ ${rk['deposit'] * rk['risk'] / 100:,.0f})").replace(",", " ")
    else:
        cur = "Депозит пока не задан."
    return ("🧮 <b>Риск и объём лота</b>\n\n" + cur + "\n\n"
            "Когда депозит задан, под каждым SL я пишу, каким объёмом входить, чтобы при стоп-лоссе потерять "
            "не больше выбранного процента депозита.\n"
            "Расчёт приблизительный: стандартный лот 100 000, депозит в долларах. Общее правило: 1% на сделку "
            "(не больше 2%).\n\nБыстро: <code>/risk 1000 1</code> (депозит 1000 $, риск 1%).")


def risk_kb():
    return {"inline_keyboard": [
        [btn("0.5%", "r:0.5"), btn("1%", "r:1"), btn("2%", "r:2")],
        [btn("✏️ Ввести депозит", "dep")],
        [btn("⬅️ Меню", "main")]]}


# ------------------------- ОПОВЕЩЕНИЯ О УРОВНЯХ -------------------------
_live = {}


def live_price(pair):
    now = time.time()
    hit = _live.get(pair)
    if hit and now - hit[0] < 240:
        return hit[1]
    df = fa.yf.download(fa.PAIRS[pair], period="1d", interval="5m", progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError("нет данных")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    price = float(df["Close"].dropna().iloc[-1])
    _live[pair] = (now, price)
    return price


def alert_text(res, t, price, rk):
    d, rule = t["dir"], RULES[t["rule"]]
    where = "выше" if t["side"] == 1 else "ниже"
    action = "Покупка" if d == 1 else "Продажа"
    sl, tp = levels(price, res["atr"], d, rule["sl"], rule["tp"])
    return "\n".join([
        f"🔔 <b>{res['pair']}</b>: цена {code(price)} дошла до уровня {code(t['level'])}",
        WAIT_WHY[(t["rule"], d)],
        f"Это ещё <b>не сигнал</b>: он появится, только если дневная свеча <b>закроется {where}</b> уровня "
        f"(около 22:00 UTC).",
        f"Если закроется: {action}, SL {code(sl)} · TP {code(tp)}{lot_suffix(res['pair'], price, sl, rk)}",
    ])


def alert_check_once():
    wanted = {}
    for chat_id, pair in q("SELECT u.chat_id, s.pair FROM users u JOIN subs s ON s.chat_id=u.chat_id "
                           "WHERE u.alerts=1"):
        wanted.setdefault(pair, []).append(chat_id)
    day = dt.datetime.now(dt.timezone.utc).date().isoformat()
    for pair, chats in wanted.items():
        try:
            res = analyze_pair(pair)
            price = live_price(pair)
        except Exception as e:
            log(f"оповещения {pair}: {e}")
            continue
        if not price or price <= 0 or abs(price - res["price"]) > 10 * res["atr"]:
            log(f"оповещения {pair}: странная цена {price}, пропускаю")
            continue
        for t in res["triggers"]:
            if not ((price >= t["level"]) if t["side"] == 1 else (price <= t["level"])):
                continue
            key = f"{t['rule']}:{t['dir']}:{t['level']:.4f}"
            for chat_id in chats:
                if q("SELECT 1 FROM alerts_sent WHERE chat_id=? AND pair=? AND key=? AND day=?",
                     (chat_id, pair, key, day)):
                    continue
                q("INSERT OR IGNORE INTO alerts_sent VALUES(?,?,?,?)", (chat_id, pair, key, day))
                send(chat_id, alert_text(res, t, price, get_risk(chat_id)))


def alert_loop():
    while True:
        try:
            if dt.datetime.now(dt.timezone.utc).weekday() < 5:
                alert_check_once()
        except Exception:
            traceback.print_exc()
        time.sleep(ALERT_EVERY)


# ------------------------- СТАТУС (для администратора) -------------------------
def status_text():
    users = q("SELECT COUNT(*) FROM users")[0][0]
    alerts_on = q("SELECT COUNT(*) FROM users WHERE alerts=1")[0][0]
    sig = q("SELECT COUNT(*), COALESCE(SUM(status='open'), 0) FROM signals")[0]
    ok = os.path.abspath(DB_PATH).startswith("/data")
    return "\n".join([
        "🛠 <b>Статус бота</b>",
        f"База: <code>{html.escape(DB_PATH)}</code>" + (" ✅" if ok else
            "\n⚠️ Путь не /data. Если это не подключённый Volume, при каждом перезапуске журнал и подписки "
            "будут стираться."),
        f"Пользователей: {users} (оповещения включены у {alerts_on})",
        f"Сигналов в журнале: {sig[0]} (открытых {sig[1]})",
        f"Последняя рассылка: {get_meta('last_daily') or 'ещё не было'}",
        f"Следующая: {DAILY_HOUR_UTC:02d}:00 UTC, по будням",
        f"Проверка оповещений: каждые {max(ALERT_EVERY // 60, 1)} мин",
    ])


def overview_text():
    lines = ["📋 <b>Обзор по всем парам</b>", ""]
    for p in fa.PAIRS:
        try:
            lines.append(one_line(analyze_pair(p, force=True)))
        except Exception as e:
            log(f"{p}: {e}")
            lines.append(f"⚠️ <b>{p}</b>: нет данных")
    return "\n".join(lines + ["", "<i>Подробности по паре: «Проверить пару».</i>"])


# ------------------------- ЖУРНАЛ СИГНАЛОВ -------------------------
def evaluate_open(data):
    rows = q("SELECT id,pair,dir,bar_date,entry,sl,tp,risk,spread FROM signals WHERE status='open'")
    for sid, pair, d, bar, entry, sl, tp, risk, spread in rows:
        try:
            if pair not in data:
                continue
            _, df = load(pair)
            sub = df[df.index.strftime("%Y-%m-%d") > bar]
            res = None
            for k, (_, row) in enumerate(sub.iterrows(), 1):
                hi, lo = row["High"], row["Low"]
                if (lo <= sl) if d == 1 else (hi >= sl):
                    res = (sl - entry) * d / risk
                elif (hi >= tp) if d == 1 else (lo <= tp):
                    res = (tp - entry) * d / risk
                elif k >= MAXHOLD:
                    res = (row["Close"] - entry) * d / risk
                if res is not None:
                    break
            if res is not None:
                q("UPDATE signals SET status='closed', r=? WHERE id=?", (res - spread / risk, sid))
        except Exception:
            traceback.print_exc()


def record_signals(data):
    for pair, res in data.items():
        for e in res["events"]:
            if e["age"] != 0:
                continue
            rule = RULES[e["rule"]]
            risk = rule["sl"] * res["atr"]
            sl, tp = levels(e["entry"], res["atr"], e["dir"], rule["sl"], rule["tp"])
            q("INSERT OR IGNORE INTO signals(pair,rule,dir,bar_date,entry,sl,tp,risk,spread) VALUES(?,?,?,?,?,?,?,?,?)",
              (pair, e["rule"], e["dir"], e["bar"], e["entry"], sl, tp, risk, SPREAD_PIPS[pair] * PIP))


def stats_text():
    rows = q("SELECT pair,rule,dir,bar_date,status,r FROM signals ORDER BY id")
    closed = [r for r in rows if r[4] == "closed"]
    opened = [r for r in rows if r[4] == "open"]
    lines = ["📊 <b>Журнал сигналов</b> (с запуска бота, результат после спреда)", ""]
    if not closed:
        lines.append("Пока нет закрытых сигналов.")
    else:
        rs = np.array([r[5] for r in closed])
        lines.append(f"Закрыто: {len(rs)} · прибыльных {np.mean(rs > 0) * 100:.0f}% · "
                     f"средний {rs.mean():+.2f}R · всего {rs.sum():+.1f}R")
        for rule, info in RULES.items():
            x = np.array([r[5] for r in closed if r[1] == rule])
            if len(x):
                lines.append(f"  • {info['title']}: {len(x)} шт., {x.mean():+.2f}R")
        lines += ["", "Последние:"]
        for pair, rule, d, bar, _, r in closed[-5:]:
            lines.append(f"  {bar} {pair} {'BUY' if d == 1 else 'SELL'}: {r:+.2f}R")
    lines.append(f"\nОткрытых сейчас: {len(opened)}")
    lines.append("R — результат в долях риска (1R = расстояние до стоп-лосса).")
    lines.append("Выборка маленькая: выводы делай после десятков сигналов.")
    return "\n".join(lines)


# ------------------------- ЕЖЕДНЕВНАЯ РАССЫЛКА -------------------------
def run_daily():
    """True, если разбор выполнен (повторный запуск в тот же день не шлёт дубли)."""
    log("ежедневный разбор: старт")
    data = {}
    for p in fa.PAIRS:
        try:
            data[p] = analyze_pair(p, force=True)
        except Exception as e:
            log(f"{p}: ошибка данных — {e}")
    if not data:
        return False
    evaluate_open(data)
    record_signals(data)
    today = dt.datetime.now(dt.timezone.utc)
    weekly = today.weekday() == 0
    day = today.date().isoformat()
    for (chat_id,) in q("SELECT chat_id FROM users WHERE daily=1"):
        mark = f"daily_sent:{day}:{chat_id}"
        if get_meta(mark) == "1":
            continue
        pairs = [r[0] for r in q("SELECT pair FROM subs WHERE chat_id=?", (chat_id,)) if r[0] in data]
        if not pairs:
            continue
        rk = get_risk(chat_id)
        sig = [p for p in pairs if any(e["age"] == 0 for e in data[p]["events"])]
        parts = [f"🌅 <b>Ежедневный разбор {today.date()}</b>"]
        for p in sig:
            parts.append(pair_text({**data[p], "events": [e for e in data[p]["events"] if e["age"] == 0]},
                                   with_triggers=False, rk=rk))
        quiet = [one_line(data[p]) for p in pairs if p not in sig]
        if quiet:
            parts.append("\n".join(quiet))
        send(chat_id, "\n\n———————\n\n".join(parts), MAIN_KB)
        if weekly:
            send(chat_id, "📅 <b>Итоги недели</b>\n\n" + stats_text())
        set_meta(mark, "1")
    log("ежедневный разбор: готово")
    return True


def scheduler():
    while True:
        try:
            now = dt.datetime.now(dt.timezone.utc)
            key = now.strftime("%Y-%m-%d")
            if now.hour >= DAILY_HOUR_UTC and get_meta("last_daily") != key:
                if now.weekday() >= 5:          # в выходные рынок закрыт
                    set_meta("last_daily", key)
                else:
                    tries = int(get_meta("tries:" + key) or 0)
                    if tries < 30:              # при сбое данных пробуем каждую минуту до 30 раз
                        set_meta("tries:" + key, str(tries + 1))
                        if run_daily():
                            set_meta("last_daily", key)
                    else:
                        set_meta("last_daily", key)
        except Exception:
            traceback.print_exc()
        time.sleep(60)


# ------------------------- ИНТЕРФЕЙС -------------------------
def btn(text, data):
    return {"text": text, "callback_data": data}


MAIN_KB = {"inline_keyboard": [
    [btn("📈 Проверить пару", "menu"), btn("📋 Обзор всех пар", "ov")],
    [btn("⭐ Мои пары", "subs"), btn("🧮 Риск и лот", "risk")],
    [btn("📊 Статистика сигналов", "stats")],
    [btn("ℹ️ Как это работает", "help")],
]}

HELP = (
    "ℹ️ Как это работает\n\n"
    "Бот смотрит дневные свечи по 5 парам и ищет сигналы по трём правилам, "
    "проверенным на 10 годах истории:\n"
    "1) цена пересекла EMA50 (основное)\n2) Stochastic вошёл в зону 20/80 (основное)\n"
    "3) разворот по RSI 30/70 (слабее)\n\n"
    "📈 «Проверить пару» — свежий расчёт: сигналы, цены, за которыми закрытие дня даст сигнал, SL/TP, новости.\n"
    "📋 «Обзор» — все пары одним экраном.\n"
    f"⭐ «Мои пары» — по каким парам получать разбор ({DAILY_HOUR_UTC:02d}:00 UTC, по будням) "
    "и включить 🔔 оповещения: бот напишет сразу, когда цена дойдёт до уровня «что ждать».\n"
    "🧮 «Риск и лот» — задай депозит и процент риска, и бот будет писать объём лота под каждый SL.\n"
    "📊 «Статистика» — как закончились прошлые сигналы бота.\n\n"
    "На истории преимущество небольшое (около +0.1–0.2R на сделку), "
    "поэтому сначала проверяй на демо-счёте. Не финансовая рекомендация."
)


def pairs_kb():
    names = list(fa.PAIRS)
    rows = [[btn(names[i], "p:" + names[i].replace("/", "")) for i in range(j, min(j + 2, len(names)))]
            for j in range(0, len(names), 2)]
    rows.append([btn("⬅️ Меню", "main")])
    return {"inline_keyboard": rows}


def subs_kb(chat_id):
    on = {r[0] for r in q("SELECT pair FROM subs WHERE chat_id=?", (chat_id,))}
    daily, alerts = q("SELECT daily, alerts FROM users WHERE chat_id=?", (chat_id,))[0]
    rows = [[btn(("✅ " if p in on else "☐ ") + p, "t:" + p.replace("/", ""))] for p in fa.PAIRS]
    rows.append([btn("🌅 Ежедневный разбор: " + ("ВКЛ" if daily else "ВЫКЛ"), "d")])
    rows.append([btn("🔔 Оповещения о уровнях: " + ("ВКЛ" if alerts else "ВЫКЛ"), "a")])
    rows.append([btn("⬅️ Меню", "main")])
    return {"inline_keyboard": rows}


SUBS_TEXT = ("⭐ Мои пары\n\nОтметь пары для ежедневного разбора и оповещений. "
             "🔔 Оповещения: бот напишет сразу, когда цена дойдёт до уровня «что ждать» "
             "(проверка раз в несколько минут по будням).")


def reply_kb():
    names = list(fa.PAIRS) + ["☰ Меню"]
    rows = [[{"text": n} for n in names[i:i + 2]] for i in range(0, len(names), 2)]
    return {"keyboard": rows, "resize_keyboard": True, "is_persistent": True}


def pair_kb(pair):
    return {"inline_keyboard": [[btn("🔄 Обновить", "p:" + pair.replace("/", ""))],
                                [btn("⬅️ К парам", "menu")]]}


def send_pair(chat_id, pair):
    r = send(chat_id, f"⏳ Считаю {pair}…")
    mid = ((r or {}).get("result") or {}).get("message_id")
    try:
        txt = pair_text(analyze_pair(pair, force=True), rk=get_risk(chat_id))
    except Exception as e:
        log(f"{pair}: {e}")
        txt = f"Не удалось получить данные по {pair}. Попробуй через минуту."
    if mid:
        edit(chat_id, mid, txt, pair_kb(pair))
    else:
        send(chat_id, txt, pair_kb(pair))


def handle_message(m):
    chat_id = m["chat"]["id"]
    name = (m.get("from") or {}).get("first_name", "")
    if ALLOWED and chat_id not in ALLOWED:
        send(chat_id, "Доступ к боту закрыт.")
        return
    new = ensure_user(chat_id, name)
    text = (m.get("text") or "").strip()
    cmd = text.split()[0].split("@")[0].lower() if text.startswith("/") else ""
    key = text.upper().replace("/", "").replace(" ", "")
    if get_meta(f"await_dep:{chat_id}") == "1" and not cmd and key not in KEY and text != "☰ Меню":
        v = parse_num(text, 10, 10_000_000)
        if v is None:
            send(chat_id, "Напиши сумму депозита числом в долларах, например 1000. Отмена: /cancel.")
        else:
            q("UPDATE users SET deposit=? WHERE chat_id=?", (v, chat_id))
            set_meta(f"await_dep:{chat_id}", "0")
            send(chat_id, risk_text(chat_id), risk_kb())
        return
    if cmd == "/start" or new:
        send(chat_id, "👋 Привет! Я слежу за 5 валютными парами (GBP/USD, EUR/USD, GBP/AUD, "
                      "EUR/NZD, EUR/CAD) и присылаю сигналы.\n\nТы уже подписан на ежедневный "
                      "разбор по всем парам. В «Моих парах» можно включить 🔔 оповещения, "
                      "а в «Риск и лот» задать депозит, и я буду считать объём лота. Что делаем?", MAIN_KB)
    elif cmd == "/help":
        send(chat_id, HELP, MAIN_KB)
    elif cmd == "/stats":
        send(chat_id, stats_text(), MAIN_KB)
    elif cmd == "/pairs":
        send(chat_id, "Выбери пару:", pairs_kb())
    elif cmd == "/overview":
        r = send(chat_id, "⏳ Считаю все пары…")
        mid = ((r or {}).get("result") or {}).get("message_id")
        txt = overview_text()
        edit(chat_id, mid, txt, MAIN_KB) if mid else send(chat_id, txt, MAIN_KB)
    elif cmd == "/risk":
        args = text.split()[1:]
        dep = parse_num(args[0], 10, 10_000_000) if args else None
        pct = parse_num(args[1], 0.1, 5) if len(args) > 1 else None
        if args and dep is None:
            send(chat_id, "Не понял сумму. Пример: /risk 1000 1 (депозит 1000 $, риск 1%).")
            return
        if dep is not None:
            q("UPDATE users SET deposit=? WHERE chat_id=?", (dep, chat_id))
        if pct is not None:
            q("UPDATE users SET risk=? WHERE chat_id=?", (pct, chat_id))
        send(chat_id, risk_text(chat_id), risk_kb())
    elif cmd == "/status":
        send(chat_id, status_text() if chat_id in ADMIN else "Команда только для администратора "
             "(задай переменную ADMIN_IDS).")
    elif cmd == "/cancel":
        set_meta(f"await_dep:{chat_id}", "0")
        send(chat_id, "Отменено.", MAIN_KB)
    elif key in KEY:
        send_pair(chat_id, KEY[key])
    else:
        send(chat_id, "Выбери действие:", MAIN_KB)
    row = q("SELECT kb FROM users WHERE chat_id=?", (chat_id,))
    if row and row[0][0] == 0:
        q("UPDATE users SET kb=1 WHERE chat_id=?", (chat_id,))
        send(chat_id, "⌨️ Внизу появилась панель с парами: нажми пару, и я сразу пришлю свежий разбор.",
             reply_kb())


def handle_callback(cb):
    chat_id = cb["message"]["chat"]["id"]
    msg_id = cb["message"]["message_id"]
    data = cb.get("data", "")
    api("answerCallbackQuery", callback_query_id=cb["id"])
    if ALLOWED and chat_id not in ALLOWED:
        return
    ensure_user(chat_id, (cb.get("from") or {}).get("first_name", ""))
    if data == "main":
        edit(chat_id, msg_id, "Выбери действие:", MAIN_KB)
    elif data == "menu":
        edit(chat_id, msg_id, "Выбери пару:", pairs_kb())
    elif data.startswith("p:") and data[2:] in KEY:
        pair = KEY[data[2:]]
        kb = {"inline_keyboard": [[btn("🔄 Обновить", data)], [btn("⬅️ К парам", "menu")]]}
        edit(chat_id, msg_id, f"⏳ Считаю {pair}…")
        try:
            edit(chat_id, msg_id, pair_text(analyze_pair(pair, force=True), rk=get_risk(chat_id)), kb)
        except Exception as e:
            log(f"{pair}: {e}")
            edit(chat_id, msg_id, f"Не удалось получить данные по {pair}. Попробуй через минуту.", kb)
    elif data == "subs":
        edit(chat_id, msg_id, SUBS_TEXT, subs_kb(chat_id))
    elif data.startswith("t:") and data[2:] in KEY:
        pair = KEY[data[2:]]
        if q("SELECT 1 FROM subs WHERE chat_id=? AND pair=?", (chat_id, pair)):
            q("DELETE FROM subs WHERE chat_id=? AND pair=?", (chat_id, pair))
        else:
            q("INSERT INTO subs VALUES(?,?)", (chat_id, pair))
        edit(chat_id, msg_id, SUBS_TEXT, subs_kb(chat_id))
    elif data == "d":
        q("UPDATE users SET daily=1-daily WHERE chat_id=?", (chat_id,))
        edit(chat_id, msg_id, SUBS_TEXT, subs_kb(chat_id))
    elif data == "a":
        q("UPDATE users SET alerts=1-alerts WHERE chat_id=?", (chat_id,))
        edit(chat_id, msg_id, SUBS_TEXT, subs_kb(chat_id))
    elif data == "ov":
        edit(chat_id, msg_id, "⏳ Считаю все пары…")
        edit(chat_id, msg_id, overview_text(),
             {"inline_keyboard": [[btn("🔄 Обновить", "ov")], [btn("⬅️ Меню", "main")]]})
    elif data == "risk":
        edit(chat_id, msg_id, risk_text(chat_id), risk_kb())
    elif data.startswith("r:"):
        pct = parse_num(data[2:], 0.1, 5)
        if pct is not None:
            q("UPDATE users SET risk=? WHERE chat_id=?", (pct, chat_id))
        edit(chat_id, msg_id, risk_text(chat_id), risk_kb())
    elif data == "dep":
        set_meta(f"await_dep:{chat_id}", "1")
        edit(chat_id, msg_id, "✏️ Напиши сумму депозита в долларах, например 1000.\nОтмена: /cancel.")
    elif data == "stats":
        edit(chat_id, msg_id, stats_text(), {"inline_keyboard": [[btn("⬅️ Меню", "main")]]})
    elif data == "help":
        edit(chat_id, msg_id, HELP, {"inline_keyboard": [[btn("⬅️ Меню", "main")]]})


def handle_update(u):
    if "message" in u:
        handle_message(u["message"])
    elif "callback_query" in u:
        handle_callback(u["callback_query"])


def setup_profile():
    """Подсказки команд и описание бота в Telegram (то, что обычно настраивают через BotFather)."""
    api("setMyCommands", commands=[
        {"command": "start", "description": "Главное меню"},
        {"command": "pairs", "description": "Проверить пару"},
        {"command": "overview", "description": "Обзор всех пар"},
        {"command": "stats", "description": "Статистика сигналов"},
        {"command": "risk", "description": "Депозит и риск: /risk 1000 1"},
        {"command": "help", "description": "Как это работает"},
    ])
    api("setMyShortDescription", short_description="Сигналы, уровни и SL/TP по 5 валютным парам. Оповещения и расчёт лота.")
    api("setMyDescription", description=(
        "Бот-аналитик по валютным парам GBP/USD, EUR/USD, GBP/AUD, EUR/NZD, EUR/CAD: "
        "ежедневный разбор, уровни «что ждать», SL/TP, важные новости, оповещения при подходе цены к уровню, "
        "расчёт объёма лота под ваш риск и журнал результатов. Не финансовая рекомендация."))


def main():
    if not TOKEN:
        raise SystemExit("Не задан TELEGRAM_BOT_TOKEN")
    init_db()
    log(f"бот запущен, база: {DB_PATH}")
    if not os.path.abspath(DB_PATH).startswith("/data"):
        log("ВНИМАНИЕ: база не в /data. Если это не подключённый Volume, данные сотрутся при перезапуске.")
    setup_profile()
    threading.Thread(target=scheduler, daemon=True).start()
    threading.Thread(target=alert_loop, daemon=True).start()
    offset = None
    while True:
        r = api("getUpdates", timeout=30, offset=offset, allowed_updates=["message", "callback_query"])
        if not r or not r.get("ok"):
            time.sleep(5)
            continue
        for u in r["result"]:
            offset = u["update_id"] + 1
            try:
                handle_update(u)
            except Exception:
                traceback.print_exc()


if __name__ == "__main__":
    main()
