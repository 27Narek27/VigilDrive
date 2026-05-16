import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models, transforms
import time
import winsound
import threading
import requests
import datetime
import io

ctk.set_appearance_mode("dark")

SERVER_URL         = "http://localhost:8000"
SOUND_THRESHOLD     = 2.0   
ALERT_THRESHOLD     = 15.0  
GRACE_PERIOD        = 1.2  

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


class VigilDriveFinal(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VigilDrive AI  ·  v8.1")
        self.geometry("1280x800")
        self.minsize(1100, 700)
        self.configure(fg_color=C_BG)

        self.DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.CLASSES = ['Drowsy', 'Non Drowsy']

        try:
            self.model = models.mobilenet_v2(weights=None)
            self.model.classifier[1] = nn.Linear(self.model.last_channel, 2)
            self.model.load_state_dict(torch.load(
                "vigil_model_v4_combined.pth", map_location=self.DEVICE, weights_only=True))
            self.model.to(self.DEVICE).eval()
        except Exception as e:
            print(f"Model error: {e}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade  = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml')

        self.is_running        = False
        self.drowsy_acc        = 0.0
        self.is_beeping        = False
        self.driver_code       = ""
        self.eyes_closed_since = None   
        self.last_open_ts      = None   
        self.eyes_closed_sec   = 0.0
        self.sound_sent        = False
        self.alert_sent        = False

        self._build_ui()

    def _sep(self, p): ctk.CTkFrame(p, height=1, fg_color=C_BORDER).pack(fill="x", padx=16, pady=8)
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
        self._lbl(sb, "Driver Monitoring System  v8.1", 9).pack(pady=(0,12))
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
        self.lbl_conf.pack(pady=(0,10))
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
            if not self.cap.isOpened(): return
            self.is_running = True
            self.btn_main.configure(text="■  STOP MONITORING",
                                    fg_color=C_RED, hover_color="#CC2244")
            self.video_label.configure(text="")
            self._reset()
            self.stream()
        else:
            self.is_running = False
            self.cap.release()
            self.btn_main.configure(text="▶  START MONITORING",
                                    fg_color=C_GREEN, hover_color="#00B87A")
            self.video_label.configure(image="", text="[ CAMERA OFFLINE ]")
            self.ring.set(1.0, C_GREEN)
            self.eye_arc.refresh(0, "open")
            self.lbl_status.configure(text="● SYSTEM IDLE", text_color=C_MUTED)
            self._reset()

    def _reset(self):
        self.drowsy_acc        = 0.0
        self.eyes_closed_since = None
        self.last_open_ts      = None
        self.eyes_closed_sec   = 0.0
        self.sound_sent        = False
        self.alert_sent        = False
        self.is_beeping        = False  

    def _play_sound(self):
        if self.is_beeping:
            return
        self.is_beeping = True
        while self.is_beeping:
            ec = self.eyes_closed_sec
            if ec < SOUND_THRESHOLD:
                break 
            if ec < 5.0:
                winsound.Beep(1800, 160)
                winsound.Beep(900,  120)
            elif ec < 10.0:
                winsound.Beep(2400, 100)
                winsound.Beep(1100,  80)
                winsound.Beep(2600, 100)
            else:
                winsound.Beep(3200,  70)
                winsound.Beep(700,   50)
                winsound.Beep(3500,  70)
                winsound.Beep(650,   50)
        self.is_beeping = False

    def detect_eyes_open(self, frame_bgr):
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(80,80))
        for (fx, fy, fw, fh) in faces:
            roi  = gray[fy:fy+fh, fx:fx+fw]
            eyes = self.eye_cascade.detectMultiScale(roi, 1.1, 10, minSize=(20,20))
            if len(eyes) > 0: return True
        return False

    def send_alert(self, frame_bgr):
        _, buf = cv2.imencode(".jpg", frame_bgr)
        bio = io.BytesIO(buf.tobytes())
        try:
            r = requests.post(f"{SERVER_URL}/alert",
                              data={"driver_code": self.driver_code},
                              files={"screenshot": ("alert.jpg", bio, "image/jpeg")},
                              timeout=15)
            if r.ok:
                n  = r.json().get("sent_to", 0)
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                self.after(0, lambda: self.lbl_alert.configure(
                    text=f"⚡ Alert sent to {n} contacts  {ts}", text_color=C_GREEN))
            else:
                self.after(0, lambda: self.lbl_alert.configure(
                    text="✗ Server error", text_color=C_RED))
        except Exception:
            self.after(0, lambda: self.lbl_alert.configure(
                text="✗ No server connection", text_color=C_RED))


    def stream(self):
        if not self.is_running: return

        ret, frame = self.cap.read()
        if not ret:
            self.after(30, self.stream)
            return

        frame   = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = self.transform(Image.fromarray(img_rgb)).unsqueeze(0).to(self.DEVICE)
        
        with torch.no_grad():
            prob       = torch.nn.functional.softmax(self.model(inp), dim=1)
            conf, pred = torch.max(prob, 1)
        
        state, confidence = self.CLASSES[pred.item()], conf.item()

        if state == 'Drowsy' and confidence > 0.65:
            self.drowsy_acc += 0.15
        else:
            self.drowsy_acc -= 0.1
        
        self.drowsy_acc = max(0.0, min(4.0, self.drowsy_acc))
        self.lbl_conf.configure(text=f"Confidence: {confidence:.0%}  [{state}]")

        eyes_open = self.detect_eyes_open(frame)
        now       = time.time()
        
        if eyes_open:
            self.last_open_ts = now
        else:
            if self.eyes_closed_since is None:
                if (self.last_open_ts is None or
                        now - self.last_open_ts >= GRACE_PERIOD):
                    self.eyes_closed_since = now
                    self.sound_sent  = False
                    self.alert_sent  = False
            if self.eyes_closed_since is not None:
                self.eyes_closed_sec = now - self.eyes_closed_since

        if not hasattr(self, '_eye_open_start'):
            self._eye_open_start = None

        if eyes_open:
            if self._eye_open_start is None:
                self._eye_open_start = now
            open_dur = now - self._eye_open_start
            if open_dur >= GRACE_PERIOD and self.eyes_closed_since is not None:
                self.eyes_closed_since = None
                self.eyes_closed_sec   = 0.0
                self.sound_sent        = False
                self.alert_sent        = False
        else:
            self._eye_open_start = None 

        if self.eyes_closed_since is not None:
            self.eyes_closed_sec = now - self.eyes_closed_since

        ec         = self.eyes_closed_sec
        eye_state  = "open" if self.eyes_closed_since is None else "closed"
        
        if ec >= SOUND_THRESHOLD and not self.sound_sent:
            self.sound_sent = True
            threading.Thread(target=self._play_sound, daemon=True).start()
            
        if ec >= ALERT_THRESHOLD and not self.alert_sent:
            self.alert_sent = True
            threading.Thread(target=self.send_alert, args=(frame.copy(),), daemon=True).start()

        self.eye_arc.refresh(ec, eye_state)

        prog = min(self.drowsy_acc / 3.0, 1.0)
        if ec >= ALERT_THRESHOLD or prog >= 0.9:
            rc, st, sc, bc = C_RED,    "⬟ WAKE UP!",     C_RED,    (255, 59, 92)
        elif ec >= SOUND_THRESHOLD or prog >= 0.5:
            rc, st, sc, bc = C_YELLOW, "◉ WARNING",       C_YELLOW, (255, 184, 0)
        else:
            rc, st, sc, bc = C_GREEN,  "● DRIVER ALERT",  C_GREEN,  (0, 229, 160)

        self.ring.set(1.0 - prog, rc)
        self.lbl_status.configure(text=st, text_color=sc)

        h, w = img_rgb.shape[:2]
        L, T = 28, 3
        for (cx2, cy2, sx, sy) in [(0,0,1,1),(w,0,-1,1),(0,h,1,-1),(w,h,-1,-1)]:
            cv2.line(img_rgb, (cx2, cy2), (cx2+sx*L, cy2), bc, T)
            cv2.line(img_rgb, (cx2, cy2), (cx2, cy2+sy*L), bc, T)
        cv2.rectangle(img_rgb, (0,0), (w-1,h-1), bc, 2)
        
        if ec > 0:
            cv2.putText(img_rgb, f"CLOSED  {ec:.1f}s",
                        (14, h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bc, 2, cv2.LINE_AA)

        img_tk = ImageTk.PhotoImage(Image.fromarray(cv2.resize(img_rgb, (820, 556))))
        self.video_label.configure(image=img_tk)
        self.video_label._image = img_tk

        self.after(10, self.stream)


if __name__ == "__main__":
    app = VigilDriveFinal()
    app.mainloop()