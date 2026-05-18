"""
VigilDrive GUI v10.0
Changes vs v9:
  - Прямоугольники вокруг глаз (Haar cascade координаты)
  - Геолокация отправляется вместе со скриншотом (geocoder, IP-based)
  - Звук НЕ останавливается пока глаза не откроются (is_beeping управляется eye_state)
"""

import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models, transforms
import time
import numpy as np
import sounddevice as sd
import threading
import requests
import datetime
import io
import geocoder
from collections import deque

ctk.set_appearance_mode("dark")

SERVER_URL      = "http://localhost:8000"
SOUND_THRESHOLD = 2.0
ALERT_THRESHOLD = 15.0
GRACE_PERIOD    = 1.2

# ── UI Colors ─────────────────────────────────────────────────────────────────
C_BG     = "#080C10"
C_PANEL  = "#0D1117"
C_CARD   = "#131920"
C_BORDER = "#1E2D3D"
C_ACCENT = "#00D4FF"
C_ACCENT2= "#0090CC"
C_GREEN  = "#00E5A0"
C_YELLOW = "#FFB800"
C_RED    = "#FF3B5C"
C_TEXT   = "#E8F4FD"
C_MUTED  = "#4A6FA5"
C_DIM    = "#1E2D3D"

MODEL_PATH = "vigil_model_v4_combined.pth"
MODEL_ARCH = "efficientnet_b0"

CONFIDENCE_THRESHOLD = 0.72
DROWSY_WINDOW        = 20
DROWSY_TRIGGER_RATIO = 0.55


# ── Geolocation (IP-based, no GPS needed) ─────────────────────────────────────
def get_location() -> dict:
    """Returns lat/lon/address or fallback if offline."""
    try:
        g = geocoder.ip("me")
        if g.ok:
            return {
                "lat":     g.latlng[0],
                "lon":     g.latlng[1],
                "address": g.address or "Unknown",
                "city":    g.city    or "",
                "country": g.country or "",
            }
    except Exception:
        pass
    return {"lat": None, "lon": None, "address": "Location unavailable",
            "city": "", "country": ""}


class RingProgressBar(ctk.CTkCanvas):
    def __init__(self, master, size=100, **kw):
        super().__init__(master, width=size, height=size,
                         bg=C_PANEL, highlightthickness=0, **kw)
        self.size = size
        self._pct  = 1.0
        self._color = C_GREEN
        self._draw()

    def set(self, pct, color):
        self._pct   = max(0.0, min(1.0, pct))
        self._color = color
        self._draw()

    def _draw(self):
        self.delete("all")
        s, cx = self.size, self.size / 2
        pad = 10
        self.create_arc(pad, pad, s-pad, s-pad, start=90, extent=-360,
                        style="arc", outline=C_DIM, width=8)
        if self._pct > 0.001:
            self.create_arc(pad, pad, s-pad, s-pad,
                            start=90, extent=-self._pct*360,
                            style="arc", outline=self._color, width=8)
        self.create_text(cx, cx-5, text=f"{int(self._pct*100)}%",
                         fill=self._color, font=("Consolas", 15, "bold"))
        self.create_text(cx, cx+11, text="SAFETY",
                         fill=C_MUTED, font=("Consolas", 8))


class EyeTimerArc(ctk.CTkCanvas):
    def __init__(self, master, size=100, **kw):
        super().__init__(master, width=size, height=size,
                         bg=C_PANEL, highlightthickness=0, **kw)
        self.size = size
        self._update(0, "open")

    def _update(self, elapsed, state):
        self.delete("all")
        s, cx = self.size, self.size / 2
        pad = 10
        pct = min(elapsed / ALERT_THRESHOLD, 1.0) if elapsed > 0 else 0

        self.create_arc(pad, pad, s-pad, s-pad, start=90, extent=-360,
                        style="arc", outline=C_DIM, width=6)

        if state == "open":
            color, label, sym = C_GREEN, "OPEN", "●"
        elif elapsed < SOUND_THRESHOLD:
            color, label, sym = C_ACCENT, f"{elapsed:.1f}s", "◑"
        elif elapsed < ALERT_THRESHOLD:
            color, label, sym = C_YELLOW, f"{elapsed:.1f}s", "◐"
        else:
            color, label, sym = C_RED, f"{elapsed:.1f}s", "○"

        if pct > 0:
            self.create_arc(pad, pad, s-pad, s-pad,
                            start=90, extent=-pct*360,
                            style="arc", outline=color, width=6)

        self.create_text(cx, cx-6, text=label,
                         fill=color, font=("Consolas", 11, "bold"))
        self.create_text(cx, cx+9, text=sym,
                         fill=color, font=("Consolas", 9))

    def refresh(self, elapsed, state):
        self._update(elapsed, state)


def _load_model(arch: str, path: str, device: torch.device):
    if arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, 2)
    else:
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(m.last_channel, 2)
    state = torch.load(path, map_location=device, weights_only=True)
    m.load_state_dict(state)
    m.to(device).eval()
    return m


class VigilDriveFinal(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VigilDrive AI  ·  v10.0")
        self.geometry("1280x800")
        self.minsize(1100, 700)
        self.configure(fg_color=C_BG)

        self.DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.CLASSES = ['Drowsy', 'Non Drowsy']

        try:
            self.model = _load_model(MODEL_ARCH, MODEL_PATH, self.DEVICE)
            print(f"✅ Model loaded: {MODEL_PATH} ({MODEL_ARCH}) on {self.DEVICE}")
        except Exception as e:
            print(f"⚠️  Model load error: {e}")
            self.model = None

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade  = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml')

        # State
        self.is_running        = False
        self.is_beeping        = False
        self.driver_code       = ""
        self.eyes_closed_since = None
        self.last_open_ts      = None
        self.eyes_closed_sec   = 0.0
        self.sound_sent        = False
        self.alert_sent        = False
        self._eye_open_start   = None
        self._drowsy_window: deque = deque(maxlen=DROWSY_WINDOW)

        # Кэшируем геолокацию при старте (IP редко меняется)
        self._location_cache   = None

        self._build_ui()

    def _sep(self, p):
        ctk.CTkFrame(p, height=1, fg_color=C_BORDER).pack(fill="x", padx=16, pady=8)

    def _lbl(self, p, t, sz=10, c=None, **kw):
        return ctk.CTkLabel(p, text=t, font=("Consolas", sz),
                            text_color=c or C_MUTED, **kw)

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._sidebar()
        self._main_area()

    def _sidebar(self):
        sb = ctk.CTkFrame(self, width=288, fg_color=C_PANEL, corner_radius=0,
                         border_width=1, border_color=C_BORDER)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.pack_propagate(False)

        lf = ctk.CTkFrame(sb, fg_color="transparent")
        lf.pack(pady=(26, 4), padx=20, fill="x")
        ctk.CTkLabel(lf, text="VIGIL", font=("Consolas", 28, "bold"),
                     text_color=C_ACCENT).pack(side="left")
        ctk.CTkLabel(lf, text="DRIVE", font=("Consolas", 28, "bold"),
                     text_color=C_TEXT).pack(side="left")
        self._lbl(sb, "Driver Monitoring System  v10.0", 9).pack(pady=(0,12))
        self._sep(sb)

        self._lbl(sb, "DRIVER CODE", 9).pack(padx=20, anchor="w")
        row = ctk.CTkFrame(sb, fg_color="transparent")
        row.pack(padx=20, pady=(4,4), fill="x")
        self.code_entry = ctk.CTkEntry(
            row, placeholder_text="ABCD-1234",
            font=("Consolas", 13), height=36,
            fg_color=C_CARD, border_color=C_BORDER, text_color=C_TEXT)
        self.code_entry.pack(side="left", fill="x", expand=True, padx=(0,6))
        ctk.CTkButton(row, text="SET", width=44, height=36,
                     font=("Consolas", 11, "bold"),
                     fg_color=C_ACCENT2, hover_color=C_ACCENT,
                     text_color=C_BG, corner_radius=8,
                     command=self.save_code).pack(side="left")
        self.lbl_code_ok = self._lbl(sb, "", 10)
        self.lbl_code_ok.pack(padx=20, anchor="w")
        self._sep(sb)

        self.btn_main = ctk.CTkButton(
            sb, text="▶  START MONITORING",
            font=("Consolas", 13, "bold"), height=48,
            fg_color=C_GREEN, hover_color="#00B87A",
            text_color=C_BG, corner_radius=12, command=self.toggle)
        self.btn_main.pack(padx=20, pady=6, fill="x")
        self._sep(sb)

        sc = ctk.CTkFrame(sb, fg_color=C_CARD, corner_radius=12,
                         border_width=1, border_color=C_BORDER)
        sc.pack(padx=20, pady=4, fill="x")
        self.lbl_status = ctk.CTkLabel(sc, text="● SYSTEM IDLE",
                                       font=("Consolas", 13, "bold"), text_color=C_MUTED)
        self.lbl_status.pack(pady=(12,4))
        self.lbl_conf = self._lbl(sc, "Confidence: —", 10)
        self.lbl_conf.pack(pady=(0,4))
        self.lbl_ratio = self._lbl(sc, "Drowsy ratio: —", 10)
        self.lbl_ratio.pack(pady=(0,4))
        self.lbl_geo = self._lbl(sc, "Location: —", 9)
        self.lbl_geo.pack(pady=(0,10))
        self._sep(sb)

        mrow = ctk.CTkFrame(sb, fg_color="transparent")
        mrow.pack(padx=20, pady=4, fill="x")
        mrow.columnconfigure((0,1), weight=1)

        for col, title, attr in [(0,"SAFETY","ring"),(1,"EYE TIMER","eye_arc")]:
            card = ctk.CTkFrame(mrow, fg_color=C_CARD, corner_radius=12,
                                border_width=1, border_color=C_BORDER)
            card.grid(row=0, column=col, padx=(0,6) if col==0 else (6,0), sticky="nsew")
            self._lbl(card, title, 8).pack(pady=(8,2))
            if attr == "ring":
                self.ring = RingProgressBar(card, size=96)
                self.ring.pack(pady=(0,8))
            else:
                self.eye_arc = EyeTimerArc(card, size=96)
                self.eye_arc.pack(pady=(0,8))
        self._sep(sb)

        tc = ctk.CTkFrame(sb, fg_color=C_CARD, corner_radius=10,
                         border_width=1, border_color=C_BORDER)
        tc.pack(padx=20, pady=4, fill="x")
        self._lbl(tc, "THRESHOLDS", 8).pack(pady=(8,4), padx=12, anchor="w")
        for icon, lbl, val, clr in [
            ("♪", "Sound alarm",  f"{SOUND_THRESHOLD:.0f}s",  C_YELLOW),
            ("⚡","Send alert",    f"{ALERT_THRESHOLD:.0f}s",  C_RED),
            ("⏱","Grace period", f"{GRACE_PERIOD:.1f}s",     C_ACCENT),
        ]:
            r2 = ctk.CTkFrame(tc, fg_color="transparent")
            r2.pack(fill="x", padx=12, pady=2)
            ctk.CTkLabel(r2, text=icon, font=("Consolas",11),
                         text_color=clr, width=18).pack(side="left")
            self._lbl(r2, lbl, 10).pack(side="left", padx=4)
            ctk.CTkLabel(r2, text=val, font=("Consolas",10,"bold"),
                         text_color=clr).pack(side="right")
        ctk.CTkFrame(tc, height=8, fg_color="transparent").pack()
        self._sep(sb)

        self.lbl_alert = self._lbl(sb, "", 10, wraplength=250)
        self.lbl_alert.pack(padx=20, pady=2)
        self.lbl_subs  = self._lbl(sb, "", 10)
        self.lbl_subs.pack(padx=20, pady=(2,14))

    def _main_area(self):
        main = ctk.CTkFrame(self, fg_color=C_BG, corner_radius=0)
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_rowconfigure(1, weight=1)
        main.grid_columnconfigure(0, weight=1)

        tb = ctk.CTkFrame(main, fg_color=C_PANEL, height=44, corner_radius=0,
                         border_width=1, border_color=C_BORDER)
        tb.grid(row=0, column=0, sticky="ew")
        tb.pack_propagate(False)
        self._lbl(tb, "LIVE FEED", 11, C_MUTED).pack(side="left", padx=20)
        self._lbl(tb, "GPU" if torch.cuda.is_available() else "CPU", 10, C_ACCENT).pack(side="right", padx=8)
        self.lbl_time = self._lbl(tb, "", 11, C_MUTED)
        self.lbl_time.pack(side="right", padx=16)

        vc = ctk.CTkFrame(main, fg_color=C_CARD, corner_radius=16,
                         border_width=1, border_color=C_BORDER)
        vc.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        self.video_label = ctk.CTkLabel(vc, text="[ CAMERA OFFLINE ]",
                                        font=("Consolas", 18), text_color=C_MUTED)
        self.video_label.pack(expand=True, fill="both", padx=2, pady=2)

        self._tick()

    def _tick(self):
        self.lbl_time.configure(
            text=datetime.datetime.now().strftime("%H:%M:%S  %d.%m.%Y"))
        self.after(1000, self._tick)

    def save_code(self):
        code = self.code_entry.get().strip().upper()
        if len(code) < 4:
            self.lbl_code_ok.configure(text="✗ Code too short", text_color=C_RED)
            return
        self.driver_code = code
        self.lbl_code_ok.configure(text=f"✓ {code}", text_color=C_GREEN)
        threading.Thread(target=self._fetch_subs, daemon=True).start()

    def _fetch_subs(self):
        try:
            r = requests.get(f"{SERVER_URL}/subscribers/{self.driver_code}", timeout=5)
            if r.ok:
                n = r.json().get("subscriber_count", 0)
                self.after(0, lambda: self.lbl_subs.configure(
                    text=f"● {n} subscriber{'s' if n!=1 else ''} connected",
                    text_color=C_ACCENT if n > 0 else C_MUTED))
        except Exception:
            pass

    def toggle(self):
        if not self.is_running:
            if not self.driver_code:
                self.lbl_code_ok.configure(text="✗ Enter code first!", text_color=C_YELLOW)
                return
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                return
            self.is_running = True
            self.btn_main.configure(text="■  STOP MONITORING",
                                    fg_color=C_RED, hover_color="#CC2244")
            self.video_label.configure(text="")
            self._reset()
            # Получаем геолокацию в фоне при старте
            threading.Thread(target=self._update_location, daemon=True).start()
            self.stream()
        else:
            self.is_running  = False
            self.is_beeping  = False   # остановить звук
            self.cap.release()
            self.btn_main.configure(text="▶  START MONITORING",
                                    fg_color=C_GREEN, hover_color="#00B87A")
            self.video_label.configure(image="", text="[ CAMERA OFFLINE ]")
            self.ring.set(1.0, C_GREEN)
            self.eye_arc.refresh(0, "open")
            self.lbl_status.configure(text="● SYSTEM IDLE", text_color=C_MUTED)
            self._reset()

    def _update_location(self):
        loc = get_location()
        self._location_cache = loc
        display = loc["city"] or loc["address"]
        self.after(0, lambda: self.lbl_geo.configure(
            text=f"📍 {display[:30]}", text_color=C_ACCENT))

    def _reset(self):
        self.eyes_closed_since = None
        self.last_open_ts      = None
        self.eyes_closed_sec   = 0.0
        self.sound_sent        = False
        self.alert_sent        = False
        self.is_beeping        = False
        self._eye_open_start   = None
        self._drowsy_window.clear()

    # ── Генерация звукового тона напрямую в аудио драйвер ──────────────────
    def _beep(self, freq: int, duration_ms: int, volume: float = 1.0):
        """
        Генерирует синусоиду и пишет напрямую в sounddevice.
        Обходит системный микшер Windows (Fn+F1 mute не влияет).
        """
        sr      = 44100
        samples = int(sr * duration_ms / 1000)
        t       = np.linspace(0, duration_ms / 1000, samples, endpoint=False)
        wave    = (volume * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        sd.play(wave, samplerate=sr)
        sd.wait()

    # ── Звук: крутится пока is_beeping=True (глаза не открылись) ─────────────
    def _play_sound(self):
        while self.is_beeping:
            ec = self.eyes_closed_sec
            if ec < 5.0:
                self._beep(1800, 160)
                self._beep(900,  120)
            elif ec < 10.0:
                self._beep(2400, 100)
                self._beep(1100,  80)
                self._beep(2600, 100)
            else:
                self._beep(3200, 70)
                self._beep(700,  50)
                self._beep(3500, 70)
                self._beep(650,  50)
            time.sleep(0.02)

    # ── Детекция глаз + рисование прямоугольников ────────────────────────────
    def detect_eyes(self, frame_bgr, img_rgb):
        """
        Рисует прямоугольники вокруг лица и глаз прямо на img_rgb.
        Возвращает: True (глаза найдены) | False (лицо есть, глаз нет) | None (нет лица)
        """
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))

        if len(faces) == 0:
            return None

        found_open = False
        for (fx, fy, fw, fh) in faces:
            # Прямоугольник вокруг лица (синий)
            cv2.rectangle(img_rgb,
                          (fx, fy), (fx + fw, fy + fh),
                          (0, 200, 255), 2)
            cv2.putText(img_rgb, "FACE",
                        (fx, fy - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 200, 255), 1, cv2.LINE_AA)

            # Ищем глаза только в верхней половине лица
            roi_gray = gray[fy : fy + fh//2, fx : fx + fw]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 8, minSize=(18, 18))

            for (ex, ey, ew, eh) in eyes:
                # Координаты в исходном кадре
                abs_x = fx + ex
                abs_y = fy + ey
                # Прямоугольник вокруг глаза (зелёный)
                cv2.rectangle(img_rgb,
                              (abs_x, abs_y), (abs_x + ew, abs_y + eh),
                              (0, 255, 160), 2)
                cv2.putText(img_rgb, "EYE",
                            (abs_x, abs_y - 4), cv2.FONT_HERSHEY_SIMPLEX,
                            0.38, (0, 255, 160), 1, cv2.LINE_AA)
                found_open = True

        return found_open if found_open else False

    # ── Отправка алерта с геолокацией ────────────────────────────────────────
    def send_alert(self, frame_bgr):
        _, buf = cv2.imencode(".jpg", frame_bgr)
        bio = io.BytesIO(buf.tobytes())

        # Если кэш пустой — запрашиваем синхронно, чтобы координаты ушли с фото
        if not self._location_cache:
            self._location_cache = get_location()

        loc     = self._location_cache
        lat     = loc.get("lat")
        lon     = loc.get("lon")
        address = loc.get("address", "Unknown")
        city    = loc.get("city", "")
        country = loc.get("country", "")

        loc_str = (
            f"Lat: {lat if lat is not None else 'N/A'}, "
            f"Lon: {lon if lon is not None else 'N/A'} | "
            f"{city}{', ' + country if country else ''} | "
            f"{address}"
        )
        maps_link = (
            f"https://www.google.com/maps?q={lat},{lon}"
            if lat is not None and lon is not None
            else "Location unavailable"
        )

        try:
            r = requests.post(
                f"{SERVER_URL}/alert",
                data={
                    "driver_code": self.driver_code,
                    "lat":         str(lat) if lat is not None else "",
                    "lon":         str(lon) if lon is not None else "",
                    "address":     address,
                    "city":        city,
                    "country":     country,
                    "location":    loc_str,
                    "maps_link":   maps_link,
                    "timestamp":   datetime.datetime.now().isoformat(),
                },
                files={"screenshot": ("alert.jpg", bio, "image/jpeg")},
                timeout=15,
            )
            if r.ok:
                n   = r.json().get("sent_to", 0)
                hts = datetime.datetime.now().strftime("%H:%M:%S")
                msg = f"⚡ Alert sent to {n} contacts  {hts}\n📍 {maps_link}"
                self.after(0, lambda: self.lbl_alert.configure(
                    text=msg, text_color=C_GREEN))
            else:
                self.after(0, lambda: self.lbl_alert.configure(
                    text=f"✗ Server error {r.status_code}", text_color=C_RED))
        except Exception as e:
            self.after(0, lambda: self.lbl_alert.configure(
                text="✗ No server connection", text_color=C_RED))
            print(f"[Alert] Error: {e}")

    # ── Main loop ─────────────────────────────────────────────────────────────
    def stream(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.after(30, self.stream)
            return

        frame   = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 1. CNN prediction
        state, confidence = "Non Drowsy", 0.0
        if self.model is not None:
            inp = self.transform(Image.fromarray(img_rgb)).unsqueeze(0).to(self.DEVICE)
            with torch.no_grad():
                prob       = torch.nn.functional.softmax(self.model(inp), dim=1)
                conf, pred = torch.max(prob, 1)
            state, confidence = self.CLASSES[pred.item()], conf.item()

        # 2. Rolling window
        is_drowsy_frame = (state == 'Drowsy' and confidence >= CONFIDENCE_THRESHOLD)
        self._drowsy_window.append(1 if is_drowsy_frame else 0)
        drowsy_ratio     = sum(self._drowsy_window) / max(len(self._drowsy_window), 1)
        drowsy_sustained = (drowsy_ratio >= DROWSY_TRIGGER_RATIO
                            and len(self._drowsy_window) >= DROWSY_WINDOW // 2)

        self.lbl_conf.configure(text=f"Confidence: {confidence:.0%}  [{state}]")
        self.lbl_ratio.configure(
            text=f"Drowsy ratio: {drowsy_ratio:.0%}",
            text_color=C_RED if drowsy_sustained else C_MUTED)

        # 3. Детекция глаз + рисование прямоугольников
        eye_result = self.detect_eyes(frame, img_rgb)

        if eye_result is True:
            eyes_open = True
        elif eye_result is None:
            eyes_open = not drowsy_sustained
        else:
            eyes_open = not drowsy_sustained

        now = time.time()

        # 4. Timer логика
        if eyes_open:
            self.last_open_ts = now
            if self._eye_open_start is None:
                self._eye_open_start = now
            open_dur = now - self._eye_open_start
            if open_dur >= GRACE_PERIOD and self.eyes_closed_since is not None:
                self.eyes_closed_since = None
                self.eyes_closed_sec   = 0.0
                self.sound_sent        = False
                self.alert_sent        = False
                # ── Останавливаем звук только когда глаза открылись ──────────
                self.is_beeping        = False
        else:
            self._eye_open_start = None
            if self.eyes_closed_since is None:
                if (self.last_open_ts is None or
                        now - self.last_open_ts >= GRACE_PERIOD):
                    self.eyes_closed_since = now
                    self.sound_sent  = False
                    self.alert_sent  = False

        if self.eyes_closed_since is not None:
            self.eyes_closed_sec = now - self.eyes_closed_since

        ec        = self.eyes_closed_sec
        eye_state = "open" if self.eyes_closed_since is None else "closed"

        # 5. Alerts
        if ec >= SOUND_THRESHOLD and not self.sound_sent:
            self.sound_sent = True
            self.is_beeping = True   # флаг: звук идёт пока глаза закрыты
            threading.Thread(target=self._play_sound, daemon=True).start()

        if ec >= ALERT_THRESHOLD and not self.alert_sent:
            self.alert_sent = True
            threading.Thread(target=self.send_alert, args=(frame.copy(),), daemon=True).start()

        self.eye_arc.refresh(ec, eye_state)

        # 6. Safety ring
        prog = drowsy_ratio
        if ec >= ALERT_THRESHOLD or prog >= 0.9:
            rc, st, sc, bc = C_RED,    "⬟ WAKE UP!",    C_RED,    (255, 59, 92)
        elif ec >= SOUND_THRESHOLD or prog >= DROWSY_TRIGGER_RATIO:
            rc, st, sc, bc = C_YELLOW, "◉ WARNING",      C_YELLOW, (255, 184, 0)
        else:
            rc, st, sc, bc = C_GREEN,  "● DRIVER ALERT", C_GREEN,  (0, 229, 160)

        self.ring.set(1.0 - prog, rc)
        self.lbl_status.configure(text=st, text_color=sc)

        # 7. Overlay на кадр
        h, w = img_rgb.shape[:2]
        L, T = 28, 3
        for (cx2, cy2, sx, sy) in [(0,0,1,1),(w,0,-1,1),(0,h,1,-1),(w,h,-1,-1)]:
            cv2.line(img_rgb, (cx2, cy2), (cx2+sx*L, cy2), bc, T)
            cv2.line(img_rgb, (cx2, cy2), (cx2, cy2+sy*L), bc, T)
        cv2.rectangle(img_rgb, (0,0), (w-1,h-1), bc, 2)

        if ec > 0:
            cv2.putText(img_rgb, f"CLOSED  {ec:.1f}s",
                        (14, h-14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, bc, 2, cv2.LINE_AA)

        ratio_color = bc if drowsy_sustained else (100, 180, 255)
        cv2.putText(img_rgb, f"Drowsy {drowsy_ratio:.0%}",
                    (14, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, ratio_color, 2, cv2.LINE_AA)

        img_tk = ImageTk.PhotoImage(Image.fromarray(cv2.resize(img_rgb, (820, 556))))
        self.video_label.configure(image=img_tk)
        self.video_label._image = img_tk

        self.after(10, self.stream)


if __name__ == "__main__":
    app = VigilDriveFinal()
    app.mainloop()