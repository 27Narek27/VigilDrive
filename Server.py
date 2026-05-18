from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3, requests, random, string

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BOT_TOKEN = "8693711674:AAEvo0P4NWUscYd2RJ7sRgIiWlWHhysz-wM"
DB_PATH   = "vigildrive.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS drivers (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                driver_code TEXT UNIQUE NOT NULL,
                chat_id     TEXT NOT NULL
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS subscribers (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                driver_code TEXT NOT NULL,
                chat_id     TEXT NOT NULL,
                UNIQUE(driver_code, chat_id)
            )
        """)
        db.commit()

init_db()

def make_code():
    chars = string.ascii_uppercase + string.digits
    while True:
        code = ''.join(random.choices(chars, k=4)) + '-' + ''.join(random.choices(chars, k=4))
        with get_db() as db:
            exists = db.execute("SELECT 1 FROM drivers WHERE driver_code=?", (code,)).fetchone()
        if not exists:
            return code

def send_telegram_photo(chat_id: str, photo_bytes: bytes, caption: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    requests.post(
        url,
        data={"chat_id": chat_id, "caption": caption, "parse_mode": "Markdown"},
        files={"photo": ("alert.jpg", photo_bytes, "image/jpeg")},
        timeout=10,
    )

def send_telegram_location(chat_id: str, lat: float, lon: float):
    """Отправляет геопозицию как интерактивную карту в Telegram."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendLocation"
    requests.post(
        url,
        json={"chat_id": chat_id, "latitude": lat, "longitude": lon},
        timeout=10,
    )

def send_telegram_text(chat_id: str, text: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(
        url,
        json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
        timeout=10,
    )

@app.post("/register")
def register_driver(name: str = Form(...), chat_id: str = Form(...)):
    code = make_code()
    with get_db() as db:
        db.execute("INSERT INTO drivers (name, driver_code, chat_id) VALUES (?,?,?)",
                   (name, code, chat_id))
        db.commit()
    send_telegram_text(chat_id,
        f"✅ *Registration Successful!*\n\n"
        f"Your code: `{code}`\n\n"
        f"Enter this code into the VigilDrive app.\n"
        f"Share this code with your loved ones — they can subscribe via the bot using:\n"
        f"`/subscribe {code}`"
    )
    return {"status": "ok", "driver_code": code}

@app.post("/subscribe")
def subscribe(driver_code: str = Form(...), chat_id: str = Form(...)):
    with get_db() as db:
        driver = db.execute("SELECT * FROM drivers WHERE driver_code=?", (driver_code,)).fetchone()
        if not driver:
            raise HTTPException(status_code=404, detail="Driver code not found")
        try:
            db.execute("INSERT INTO subscribers (driver_code, chat_id) VALUES (?,?)",
                       (driver_code, chat_id))
            db.commit()
        except sqlite3.IntegrityError:
            pass
    send_telegram_text(chat_id,
        f"✅ You are now subscribed to notifications for driver *{driver['name']}*.\n"
        f"You will receive a photo and location if the driver appears to be in distress."
    )
    return {"status": "ok", "driver_name": driver["name"]}

@app.post("/alert")
async def alert(
    driver_code: str = Form(...),
    screenshot:  UploadFile = File(...),
    # ── Координаты от клиента ──────────────────────────────────────────────
    lat:         str = Form(default=""),
    lon:         str = Form(default=""),
    address:     str = Form(default=""),
    city:        str = Form(default=""),
    country:     str = Form(default=""),
    maps_link:   str = Form(default=""),
    timestamp:   str = Form(default=""),
):
    with get_db() as db:
        driver = db.execute("SELECT * FROM drivers WHERE driver_code=?",
                            (driver_code,)).fetchone()
        if not driver:
            raise HTTPException(status_code=404, detail="Driver not found")
        subs = db.execute("SELECT chat_id FROM subscribers WHERE driver_code=?",
                          (driver_code,)).fetchall()

    photo_bytes = await screenshot.read()

    # ── Парсим координаты ─────────────────────────────────────────────────
    try:
        lat_f = float(lat) if lat else None
        lon_f = float(lon) if lon else None
    except ValueError:
        lat_f = lon_f = None

    # ── Собираем подпись к фото ───────────────────────────────────────────
    location_line = ""
    if maps_link and maps_link != "Location unavailable":
        location_line = f"\n📍 [Открыть на карте]({maps_link})"
    elif lat_f and lon_f:
        location_line = f"\n📍 Координаты: `{lat_f:.5f}, {lon_f:.5f}`"
    else:
        location_line = "\n📍 Геолокация недоступна"

    if city or country:
        location_line += f"\n🏙 {city}{', ' + country if country else ''}"

    if timestamp:
        try:
            from datetime import datetime
            ts = datetime.fromisoformat(timestamp).strftime("%H:%M:%S  %d.%m.%Y")
        except Exception:
            ts = timestamp
        location_line += f"\n🕐 {ts}"

    caption = (
        f"🚨 *VIGIL DRIVE — EMERGENCY ALERT* 🚨\n\n"
        f"Водитель *{driver['name']}* не открывал глаза более 15 секунд!\n"
        f"Немедленно свяжитесь с ним!"
        f"{location_line}"
    )

    sent = 0
    for sub in subs:
        cid = sub["chat_id"]
        try:
            # 1. Отправляем фото с подписью и ссылкой
            send_telegram_photo(cid, photo_bytes, caption)
            # 2. Отправляем интерактивную геопозицию (карта прямо в чате)
            if lat_f is not None and lon_f is not None:
                send_telegram_location(cid, lat_f, lon_f)
            sent += 1
        except Exception as e:
            print(f"Failed to send to {cid}: {e}")

    return {"status": "ok", "sent_to": sent}

@app.get("/subscribers/{driver_code}")
def get_subscribers(driver_code: str):
    with get_db() as db:
        count = db.execute(
            "SELECT COUNT(*) as c FROM subscribers WHERE driver_code=?",
            (driver_code,)
        ).fetchone()["c"]
    return {"driver_code": driver_code, "subscriber_count": count}