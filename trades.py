#!/usr/bin/env python3
"""
Интрадей-сигналы и слежение за сделкой в Telegram (без ПК).

Как работает:
  1) scan_loop: раз в час после закрытия 1H-свечи ищет решительный сигнал (intraday.py)
     и рассылает его: вход / SL / TP / время жизни сделки.
  2) Кнопка «Я вошёл» запускает слежение: track_loop раз в минуту смотрит минутные свечи и пишет:
     половина пути к TP (с кнопкой «SL в безубыток»), цена близко к стопу, TP, SL,
     скорая важная новость, 30 минут до конца окна, время вышло.
  3) Все сигналы бота (даже если никто не вошёл) ведутся «виртуальной» сделкой chat_id=0:
     из неё считается честная живая статистика интрадей-режима.

Модуль подключается к боту через bind(): так он использует базу, send/edit и кнопки бота.
"""
import datetime as dt
import html
import time
import traceback

import pandas as pd

import forex_analyzer as fa
import intraday as idy

B = None            # модуль бота (устанавливается через bind)
PIP = idy.PIP
GRADE_TEXT = {"A": "все фильтры и 3+ подтверждения", "B": "все фильтры и 2 подтверждения",
              "C": "все фильтры, мало подтверждений"}
COLS = ("id", "chat_id", "pair", "dir", "entry", "sl", "tp", "sl0", "opened", "expires",
        "status", "last_check", "flags", "close_price", "r", "sig_id")


def bind(bot_module):
    global B
    B = bot_module


def init_tables():
    B.q("""CREATE TABLE IF NOT EXISTS isignals(id INTEGER PRIMARY KEY AUTOINCREMENT, pair TEXT, dir INTEGER,
        grade TEXT, rules TEXT, bar_ts TEXT, entry REAL, sl REAL, tp REAL, atr REAL, bar_end REAL, expires REAL,
        created REAL, UNIQUE(pair, bar_ts))""")
    B.q("""CREATE TABLE IF NOT EXISTS trades(id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER, pair TEXT,
        dir INTEGER, entry REAL, sl REAL, tp REAL, sl0 REAL, opened REAL, expires REAL,
        status TEXT DEFAULT 'open', last_check REAL, flags TEXT DEFAULT '', close_price REAL, r REAL,
        sig_id INTEGER)""")
    try:
        B.q("ALTER TABLE users ADD COLUMN intraday INTEGER DEFAULT 1")
    except Exception:
        pass


# ------------------------- ДАННЫЕ -------------------------
_c1m = {}


def bars_1m(pair):
    hit = _c1m.get(pair)
    if hit and time.time() - hit[0] < 40:
        return hit[1]
    df = fa.yf.download(fa.PAIRS[pair], period="2d", interval="1m", progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError("нет данных")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = idy.to_utc(df[["Open", "High", "Low", "Close"]].dropna())
    _c1m[pair] = (time.time(), df)
    return df


def code(x):
    return f"<code>{x:.5f}</code>"


def trade_row(r):
    return dict(zip(COLS, r))


def get_trade(tid):
    rows = B.q(f"SELECT {','.join(COLS)} FROM trades WHERE id=?", (tid,))
    return trade_row(rows[0]) if rows else None


def has_flag(t, f):
    return f in (t["flags"] or "").split(",")


def add_flag(t, f):
    t["flags"] = ((t["flags"] or "") + "," + f).strip(",")
    B.q("UPDATE trades SET flags=? WHERE id=?", (t["flags"], t["id"]))


# ------------------------- СООБЩЕНИЯ -------------------------
def trade_kb(tid, be=True):
    rows = [[B.btn("🔄 Обновить", f"tr:{tid}")]]
    if be:
        rows.append([B.btn("🛡 SL в безубыток", f"be:{tid}")])
    rows.append([B.btn("🚪 Закрыл сам", f"cl:{tid}")])
    return {"inline_keyboard": rows}


def metrics(t, last):
    d = t["dir"]
    risk0 = abs(t["entry"] - t["sl0"])
    pnl = (last - t["entry"]) * d
    return dict(pips=pnl / PIP, r=pnl / risk0 if risk0 else 0.0,
                to_tp=(t["tp"] - last) * d / PIP, to_sl=(last - t["sl"]) * d / PIP,
                prog=(pnl / abs(t["tp"] - t["entry"])) if t["tp"] != t["entry"] else 0.0)


def scale(t, last, width=12):
    lo, hi = sorted((t["sl"], t["tp"]))
    pos = 0 if hi == lo else min(max((last - lo) / (hi - lo), 0), 1)
    k = round(pos * (width - 1))
    bar = "─" * k + "●" + "─" * (width - 1 - k)
    left, right = ("SL", "TP") if t["dir"] == 1 else ("TP", "SL")
    return f"{left} ├{bar}┤ {right}"


def left_text(t):
    m = max(int((t["expires"] - time.time()) / 60), 0)
    return f"{m // 60} ч {m % 60:02d} мин"


def status_text(t, last):
    m = metrics(t, last)
    d = t["dir"]
    be = abs(t["sl"] - t["entry"]) < PIP * 0.2
    return "\n".join([
        f"📍 <b>{t['pair']}</b> · {'🟢 покупка' if d == 1 else '🔴 продажа'} · вход {code(t['entry'])}",
        f"Сейчас {code(last)} · <b>{m['pips']:+.1f} п.</b> ({m['r']:+.2f}R)",
        scale(t, last),
        f"До TP {m['to_tp']:.1f} п. · до SL {m['to_sl']:.1f} п."
        + (" (стоп в безубытке)" if be else ""),
        f"⏱ Осталось {left_text(t)}",
    ])


def signal_text(s, rk=None):
    d = s["dir"]
    until = dt.datetime.fromtimestamp(s["expires"], dt.timezone.utc).strftime("%H:%M")
    lot = B.lot_suffix(s["pair"], s["entry"], s["sl"], rk)
    note = idy.BT_NOTE or "⚠️ бэктест интрадей-режима пока не подтверждён: сначала проверь на демо-счёте"
    return "\n".join([
        f"{'🟢 <b>ПОКУПКА</b>' if d == 1 else '🔴 <b>ПРОДАЖА</b>'} {s['pair']} · класс <b>{s['grade']}</b>",
        "",
        f"Вход: {code(s['entry'])} (по рынку, сейчас)",
        f"SL: {code(s['sl'])} ({abs(s['entry'] - s['sl']) / PIP:.0f} п.)",
        f"TP: {code(s['tp'])} ({abs(s['tp'] - s['entry']) / PIP:.0f} п.){lot}",
        f"⏱ Держим не дольше 4 часов, до ~{until} UTC",
        "",
        "Почему:", *[f"• {html.escape(w)}" for w in s["why"]],
        "",
        f"Класс {s['grade']}: {GRADE_TEXT[s['grade']]}.",
        note,
        "Нажми «Я вошёл», и я буду следить за сделкой. Поставь SL и TP ордерами у брокера.",
    ])


# ------------------------- ЗАКРЫТИЕ -------------------------
def close_trade(t, price, why, note=""):
    d = t["dir"]
    risk0 = abs(t["entry"] - t["sl0"])
    r = (price - t["entry"]) * d / risk0 if risk0 else 0.0
    if t["chat_id"] == 0:       # виртуальная сделка журнала: результат после спреда
        r -= idy.SPREAD_PIPS.get(t["pair"], 1.5) * PIP / risk0 if risk0 else 0.0
    B.q("UPDATE trades SET status=?, close_price=?, r=? WHERE id=?", (why, price, r, t["id"]))
    t["status"] = why
    if t["chat_id"] == 0:
        return
    pips = (price - t["entry"]) * d / PIP
    head = {"tp": "🎯 <b>Тейк-профит достигнут</b>", "sl": "🛑 <b>Стоп-лосс сработал</b>",
            "be": "🛡 <b>Закрыто в безубытке</b>", "time": "⏰ <b>Время сделки вышло: закрывай по рынку</b>",
            "manual": "🚪 <b>Сделка закрыта</b>"}[why]
    tail = {"tp": "Если TP стоял ордером, сделка уже закрыта. Иначе закрой вручную.",
            "sl": "Если SL стоял ордером, сделка уже закрыта. Иначе закрой вручную. Ждём следующий сигнал.",
            "be": "Цена вернулась ко входу после переноса стопа.",
            "time": "Прошло 4 часа: идея не отработала, держать дальше нет смысла.",
            "manual": ""}[why]
    B.send(t["chat_id"], "\n".join(x for x in [
        f"{head} · {t['pair']}", f"Вход {code(t['entry'])} → выход {code(price)}",
        f"Результат: <b>{pips:+.1f} п.</b> ({r:+.2f}R)", note, tail] if x))


# ------------------------- СЛЕЖЕНИЕ -------------------------
def process(t, df):
    now = time.time()
    d = t["dir"]
    last = float(df["Close"].iloc[-1])
    since = pd.Timestamp(t["last_check"], unit="s", tz="UTC")
    sub = df[df.index >= since]
    hi = float(sub["High"].max()) if len(sub) else last
    lo = float(sub["Low"].min()) if len(sub) else last
    hit_sl = (lo <= t["sl"]) if d == 1 else (hi >= t["sl"])
    hit_tp = (hi >= t["tp"]) if d == 1 else (lo <= t["tp"])
    if hit_sl:
        be = abs(t["sl"] - t["entry"]) < PIP * 0.2
        note = "В одном окне задеты и SL, и TP: считаю по худшему (SL)." if hit_tp else ""
        close_trade(t, t["sl"], "be" if be else "sl", note)
        return
    if hit_tp:
        close_trade(t, t["tp"], "tp")
        return
    B.q("UPDATE trades SET last_check=? WHERE id=?", (df.index[-1].timestamp(), t["id"]))
    if t["chat_id"] == 0:
        if now >= t["expires"]:
            close_trade(t, last, "time")
        return
    m = metrics(t, last)
    if now >= t["expires"]:
        close_trade(t, last, "time")
        return
    kb = trade_kb(t["id"], be=abs(t["sl"] - t["entry"]) >= PIP * 0.2)
    if m["prog"] >= 0.5 and not has_flag(t, "half"):
        add_flag(t, "half")
        B.send(t["chat_id"], "🟡 <b>Цена прошла половину пути к TP.</b> Можно перенести SL в безубыток, "
               "чтобы сделка уже не могла уйти в минус.\n\n" + status_text(t, last), kb)
    elif m["to_sl"] <= 0.2 * abs(t["entry"] - t["sl0"]) / PIP and not has_flag(t, "nearsl"):
        add_flag(t, "nearsl")
        B.send(t["chat_id"], f"⚠️ <b>Цена близко к стопу</b> (до SL {m['to_sl']:.1f} п.). Ничего не двигай: "
               "стоп нужен именно для такого случая.\n\n" + status_text(t, last), kb)
    elif t["expires"] - now <= 1800 and not has_flag(t, "t30"):
        add_flag(t, "t30")
        B.send(t["chat_id"], "⏳ <b>До конца окна сделки 30 минут.</b> Если к концу идея не отработает, "
               "закрой по рынку.\n\n" + status_text(t, last), kb)
    base, quote = t["pair"].split("/")
    for h, c, title in fa.upcoming_events({base, quote}, hours=0.5)[:1]:
        key = "news:" + html.escape(title)[:20].replace(",", " ")
        if not has_flag(t, key):
            add_flag(t, key)
            B.send(t["chat_id"], f"📰 <b>Через {h * 60:.0f} мин важная новость:</b> {c} {html.escape(title)}. "
                   "Спред и скачки цены вырастут: подумай о безубытке или закрытии.\n\n"
                   + status_text(t, last), kb)


def track_once():
    rows = B.q(f"SELECT {','.join(COLS)} FROM trades WHERE status='open'")
    by_pair = {}
    for r in rows:
        t = trade_row(r)
        by_pair.setdefault(t["pair"], []).append(t)
    for pair, ts in by_pair.items():
        try:
            df = bars_1m(pair)
        except Exception as e:
            B.log(f"слежение {pair}: {e}")
            continue
        for t in ts:
            try:
                process(t, df)
            except Exception:
                traceback.print_exc()


def track_loop():
    while True:
        try:
            if dt.datetime.now(dt.timezone.utc).weekday() < 5:
                track_once()
        except Exception:
            traceback.print_exc()
        time.sleep(60)


# ------------------------- ПОИСК СИГНАЛОВ -------------------------
def scan_once():
    now = pd.Timestamp.now(tz="UTC")
    if now.weekday() >= 5 or now.minute < 2 or now.minute > 25:
        return
    hour_key = now.strftime("%Y-%m-%d %H")
    for pair in fa.PAIRS:
        if B.get_meta(f"iscan:{pair}") == hour_key:
            continue
        try:
            s, fresh = idy.latest_signal(pair, now=now)
        except Exception as e:
            B.log(f"интрадей {pair}: {e}")
            continue
        if not fresh:                  # Yahoo ещё не отдал новую свечу: попробуем через минуту
            continue
        B.set_meta(f"iscan:{pair}", hour_key)
        if not s or s["pts"] < idy.MIN_PTS:
            continue
        base, quote = pair.split("/")
        if fa.upcoming_events({base, quote}, hours=idy.NEWS_BLOCK_HOURS):
            B.log(f"интрадей {pair}: сигнал отклонён из-за новостей")
            continue
        if B.q("SELECT 1 FROM isignals WHERE pair=? AND bar_ts=?", (pair, s["bar_ts"])):
            continue
        B.q("INSERT INTO isignals(pair,dir,grade,rules,bar_ts,entry,sl,tp,atr,bar_end,expires,created) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (pair, s["dir"], s["grade"], ",".join(s["rules"]), s["bar_ts"], s["entry"], s["sl"], s["tp"],
             s["atr"], s["bar_end"], s["expires"], time.time()))
        sid = B.q("SELECT id FROM isignals WHERE pair=? AND bar_ts=?", (pair, s["bar_ts"]))[0][0]
        B.q("INSERT INTO trades(chat_id,pair,dir,entry,sl,tp,sl0,opened,expires,last_check,sig_id) "
            "VALUES(0,?,?,?,?,?,?,?,?,?,?)",
            (pair, s["dir"], s["entry"], s["sl"], s["tp"], s["sl"], s["bar_end"], s["expires"], s["bar_end"], sid))
        kb = {"inline_keyboard": [[B.btn("✅ Я вошёл", f"in:{sid}"), B.btn("⏭ Пропустить", "main")]]}
        for (chat_id,) in B.q("SELECT u.chat_id FROM users u JOIN subs s ON s.chat_id=u.chat_id "
                              "WHERE u.intraday=1 AND s.pair=?", (pair,)):
            B.send(chat_id, signal_text(s, B.get_risk(chat_id)), kb)
        B.log(f"интрадей: сигнал {pair} {s['dir']} класс {s['grade']}")


def scan_loop():
    while True:
        try:
            scan_once()
        except Exception:
            traceback.print_exc()
        time.sleep(60)


# ------------------------- КНОПКИ -------------------------
def open_trade(chat_id, sid):
    rows = B.q("SELECT pair,dir,entry,sl,tp,bar_end,expires FROM isignals WHERE id=?", (sid,))
    if not rows:
        return "Сигнал не найден.", None
    pair, d, entry, sl, tp, bar_end, expires = rows[0]
    if B.q("SELECT 1 FROM trades WHERE chat_id=? AND sig_id=? AND status='open'", (chat_id, sid)):
        return "Эту сделку я уже веду. Смотри «📍 Мои сделки».", None
    now = time.time()
    if now >= expires or now - bar_end > 40 * 60:
        return "⌛ Сигнал устарел: прошло слишком много времени. Жди следующий.", None
    try:
        price = float(bars_1m(pair)["Close"].iloc[-1])
    except Exception:
        return "Не получил цену. Попробуй ещё раз через минуту.", None
    risk = abs(entry - sl)
    if (price - entry) * d < -0.5 * risk:
        return ("⌛ Цена уже ушла против сигнала больше чем на половину стопа: вход потерял смысл. "
                "Жди следующий сигнал."), None
    if (price - entry) * d > 0.5 * abs(tp - entry):
        return "⌛ Цена уже прошла половину пути к TP: входить поздно. Жди следующий сигнал.", None
    B.q("INSERT INTO trades(chat_id,pair,dir,entry,sl,tp,sl0,opened,expires,last_check,sig_id) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?)", (chat_id, pair, d, price, sl, tp, sl, now, expires, now - 60, sid))
    tid = B.q("SELECT MAX(id) FROM trades WHERE chat_id=?", (chat_id,))[0][0]
    t = get_trade(tid)
    return ("✅ <b>Слежу за сделкой.</b> Напишу, когда цена пройдёт половину пути, подойдёт к стопу, "
            "достигнет TP/SL, перед новостью и когда выйдет время.\n\n" + status_text(t, price)), trade_kb(tid)


def my_trades_text(chat_id):
    rows = B.q(f"SELECT {','.join(COLS)} FROM trades WHERE chat_id=? AND status='open' ORDER BY id", (chat_id,))
    opened = [trade_row(r) for r in rows]
    parts = []
    for t in opened:
        try:
            parts.append(status_text(t, float(bars_1m(t["pair"])["Close"].iloc[-1])))
        except Exception:
            parts.append(f"📍 {t['pair']} · цену сейчас получить не удалось")
    hist = B.q("SELECT pair,dir,r,status FROM trades WHERE chat_id=? AND status!='open' ORDER BY id DESC LIMIT 5",
               (chat_id,))
    lines = ["📍 <b>Мои сделки</b>", ""]
    lines += ["\n\n".join(parts)] if parts else ["Открытых сделок нет. Когда придёт сигнал, нажми «Я вошёл»."]
    if hist:
        lines += ["", "<b>Последние закрытые:</b>"]
        lines += [f"  {p} {'BUY' if d == 1 else 'SELL'}: {r:+.2f}R ({st})" for p, d, r, st in hist]
    return "\n".join(lines), opened


def intraday_stats_text():
    rows = B.q("SELECT pair,dir,r FROM trades WHERE chat_id=0 AND status!='open' ORDER BY id")
    lines = ["", "⚡ <b>Интрадей-сигналы</b> (все сигналы бота, живая проверка после спреда)"]
    if not rows:
        lines.append("Пока нет закрытых сигналов.")
        return "\n".join(lines)
    rs = [r[2] for r in rows]
    lines.append(f"Закрыто: {len(rs)} · прибыльных {100 * sum(x > 0 for x in rs) / len(rs):.0f}% · "
                 f"средний {sum(rs) / len(rs):+.2f}R · всего {sum(rs):+.1f}R")
    for p, d, r in rows[-5:]:
        lines.append(f"  {p} {'BUY' if d == 1 else 'SELL'}: {r:+.2f}R")
    lines.append("Выборка маленькая: выводы делай после нескольких десятков сигналов.")
    return "\n".join(lines)


def handle_callback(chat_id, msg_id, data):
    """True, если кнопка обработана этим модулем."""
    if data.startswith("in:"):
        txt, kb = open_trade(chat_id, int(data[3:]))
        B.send(chat_id, txt, kb)
        return True
    if data == "mytr":
        txt, _ = my_trades_text(chat_id)
        B.edit(chat_id, msg_id, txt, {"inline_keyboard": [[B.btn("🔄 Обновить", "mytr")],
                                                           [B.btn("⬅️ Меню", "main")]]})
        return True
    if data == "i":
        B.q("UPDATE users SET intraday=1-intraday WHERE chat_id=?", (chat_id,))
        B.edit(chat_id, msg_id, B.SUBS_TEXT, B.subs_kb(chat_id))
        return True
    if data[:3] in ("tr:", "be:", "cl:"):
        t = get_trade(int(data[3:]))
        if not t or t["chat_id"] != chat_id:
            return True
        if t["status"] != "open":
            B.edit(chat_id, msg_id, "Эта сделка уже закрыта.", {"inline_keyboard": [[B.btn("⬅️ Меню", "main")]]})
            return True
        try:
            last = float(bars_1m(t["pair"])["Close"].iloc[-1])
        except Exception:
            B.send(chat_id, "Не получил цену. Попробуй через минуту.")
            return True
        if data.startswith("be:"):
            if (last - t["entry"]) * t["dir"] <= 0:
                B.send(chat_id, "Безубыток ставят, когда цена уже в плюсе. Сейчас сделка не в прибыли.")
                return True
            B.q("UPDATE trades SET sl=? WHERE id=?", (t["entry"], t["id"]))
            t["sl"] = t["entry"]
            B.edit(chat_id, msg_id, "🛡 Стоп перенесён в безубыток. Не забудь изменить SL и у брокера.\n\n"
                   + status_text(t, last), trade_kb(t["id"], be=False))
        elif data.startswith("cl:"):
            close_trade(t, last, "manual")
            B.edit(chat_id, msg_id, "Сделка закрыта.", {"inline_keyboard": [[B.btn("⬅️ Меню", "main")]]})
        else:
            B.edit(chat_id, msg_id, status_text(t, last),
                   trade_kb(t["id"], be=abs(t["sl"] - t["entry"]) >= PIP * 0.2))
        return True
    return False
