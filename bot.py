import telebot
import requests
BOT_TOKEN   = "8693711674:AAEvo0P4NWUscYd2RJ7sRgIiWlWHhysz-wM"     
SERVER_URL  = "http://localhost:8000"

bot = telebot.TeleBot(BOT_TOKEN)

awaiting_name = {}

@bot.message_handler(commands=["start"])
def cmd_start(msg):
    bot.send_message(msg.chat.id,
        "👋 Welcome to *VigilDrive* bot!\n\n"
        "If you are a *driver*:\n"
        "  /register — register and get your unique code\n\n"
        "If you are a *relative/friend*:\n"
        "  /subscribe CODE — subscribe to driver's alerts\n"
        "  (The driver must provide you with their code)",
        parse_mode="Markdown"
    )

@bot.message_handler(commands=["register"])
def cmd_register(msg):
    awaiting_name[msg.chat.id] = True
    bot.send_message(msg.chat.id,
        "📝 Please enter your name (it will be shown in alerts to your relatives):"
    )

@bot.message_handler(func=lambda msg: msg.chat.id in awaiting_name)
def handle_name(msg):
    name = msg.text.strip()
    chat_id = str(msg.chat.id)
    del awaiting_name[msg.chat.id]

    try:
        resp = requests.post(f"{SERVER_URL}/register",
                             data={"name": name, "chat_id": chat_id}, timeout=10)
        if resp.ok:
            code = resp.json()["driver_code"]
            bot.send_message(msg.chat.id,
                f"✅ *Success, {name}!*\n\n"
                f"Your code: `{code}`\n\n"
                f"1️⃣ Enter this code in the VigilDrive app.\n"
                f"2️⃣ Give this code to your relatives — they should type:\n"
                f"   `/subscribe {code}`",
                parse_mode="Markdown"
            )
        else:
            bot.send_message(msg.chat.id, "❌ Server error. Please try again later.")
    except Exception as e:
        bot.send_message(msg.chat.id, f"❌ Failed to connect to server: {e}")

@bot.message_handler(commands=["subscribe"])
def cmd_subscribe(msg):
    parts = msg.text.strip().split()
    if len(parts) < 2:
        bot.send_message(msg.chat.id,
            "❗ Please provide a driver code:\n`/subscribe CODE`",
            parse_mode="Markdown"
        )
        return

    driver_code = parts[1].upper()
    chat_id     = str(msg.chat.id)

    try:
        resp = requests.post(f"{SERVER_URL}/subscribe",
                             data={"driver_code": driver_code, "chat_id": chat_id}, timeout=10)
        if resp.ok:
            driver_name = resp.json().get("driver_name", "")
            bot.send_message(msg.chat.id,
                f"✅ You are now subscribed to alerts for driver *{driver_name}*.\n"
                f"If they are in trouble, you will receive a photo and a message.",
                parse_mode="Markdown"
            )
        elif resp.status_code == 404:
            bot.send_message(msg.chat.id,
                "❌ Code not found. Please double-check the code.")
        else:
            bot.send_message(msg.chat.id, "❌ Server error. Please try again later.")
    except Exception as e:
        bot.send_message(msg.chat.id, f"❌ Failed to connect to server: {e}")

@bot.message_handler(commands=["mystatus"])
def cmd_status(msg):
    bot.send_message(msg.chat.id,
        "Open the VigilDrive app — your status is displayed there."
    )

print("🤖 VigilDrive bot is running...")
bot.infinity_polling()