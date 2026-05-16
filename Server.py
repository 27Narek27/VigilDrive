from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3, requests, random, string, os

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

def make_code(length=8):
    chars = string.ascii_uppercase + string.digits
    while True:
        # Generates a code like ABCD-1234
        code = ''.join(random.choices(chars, k=4)) + '-' + ''.join(random.choices(chars, k=4))
        with get_db() as db:
            exists = db.execute("SELECT 1 FROM drivers WHERE driver_code=?", (code,)).fetchone()
        if not exists:
            return code

def send_telegram_photo(chat_id: str, photo_bytes: bytes, caption: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"
    requests.post(url, data={"chat_id": chat_id, "caption": caption, "parse_mode": "Markdown"},
                  files={"photo": ("alert.jpg", photo_bytes, "image/jpeg")}, timeout=10)

def send_telegram_text(chat_id: str, text: str):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=10)

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
        f"You will receive a photo and a message if the driver appears to be in distress."
    )
    return {"status": "ok", "driver_name": driver["name"]}


@app.post("/alert")
async def alert(driver_code: str = Form(...), screenshot: UploadFile = File(...)):
    with get_db() as db:
        driver = db.execute("SELECT * FROM drivers WHERE driver_code=?", (driver_code,)).fetchone()
        if not driver:
            raise HTTPException(status_code=404, detail="Driver not found")
        
        subs = db.execute("SELECT chat_id FROM subscribers WHERE driver_code=?",
                          (driver_code,)).fetchall()

    photo_bytes = await screenshot.read()
    caption = (
        f"🚨 *VIGIL DRIVE — EMERGENCY ALERT* 🚨\n\n"
        f"Driver *{driver['name']}* has not opened their eyes for more than 15 seconds!\n"
        f"Please contact them immediately!"
    )

    sent = 0
    for sub in subs:
        try:
            send_telegram_photo(sub["chat_id"], photo_bytes, caption)
            sent += 1
        except Exception as e:
            print(f"Failed to send to {sub['chat_id']}: {e}")

    return {"status": "ok", "sent_to": sent}


@app.get("/subscribers/{driver_code}")
def get_subscribers(driver_code: str):
    with get_db() as db:
        count = db.execute("SELECT COUNT(*) as c FROM subscribers WHERE driver_code=?",
                           (driver_code,)).fetchone()["c"]
    return {"driver_code": driver_code, "subscriber_count": count}