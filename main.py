import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
import torch
import torch.nn as nn
from torchvision import models, transforms
import time
import winsound
import threading

ctk.set_appearance_mode("dark")

class VigilDriveFinal(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("VigilDrive AI v7.0 - Gold Edition")
        self.geometry("1100x750")
        self.configure(fg_color="#0f0f0f") 

        
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.CLASSES = ['Drowsy', 'Non Drowsy']
        
    
        try:
            self.model = models.mobilenet_v2(weights=None)
            self.model.classifier[1] = nn.Linear(self.model.last_channel, 2)
            self.model.load_state_dict(torch.load("vigil_model_v4_combined.pth", map_location=self.DEVICE, weights_only=True))
            self.model.to(self.DEVICE).eval()
        except Exception as e:
            print(f"Критическая ошибка: Положите файл модели в папку с кодом! {e}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    
        self.is_running = False
        self.drowsy_acc = 0.0      
        self.last_frame_time = time.time()
        self.is_beeping = False

    
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=280, fg_color="#1a1a1a", corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        self.logo = ctk.CTkLabel(self.sidebar, text="VIGIL DRIVE", font=("Orbitron", 32, "bold"), text_color="#00ffcc")
        self.logo.pack(pady=50)

        self.btn_main = ctk.CTkButton(self.sidebar, text="START MONITORING", font=("Arial", 16, "bold"),
                                      height=55, fg_color="#00ffcc", text_color="#000",
                                      hover_color="#00cca3", corner_radius=15, command=self.toggle)
        self.btn_main.pack(pady=20, padx=30)

        self.status_box = ctk.CTkFrame(self.sidebar, fg_color="#252525", corner_radius=10)
        self.status_box.pack(fill="x", padx=20, pady=20)
        
        self.lbl_status = ctk.CTkLabel(self.status_box, text="SYSTEM IDLE", font=("Arial", 14, "bold"), text_color="gray")
        self.lbl_status.pack(pady=10)

        self.progress = ctk.CTkProgressBar(self.sidebar, width=220, height=12, fg_color="#333", progress_color="#00ffcc")
        self.progress.set(0)
        self.progress.pack(pady=10)

        self.lbl_info = ctk.CTkLabel(self.sidebar, text="Safety Level: 100%", font=("Arial", 12))
        self.lbl_info.pack()

        # Видео-панель
        self.view_area = ctk.CTkFrame(self, fg_color="#141414", corner_radius=20)
        self.view_area.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.video_label = ctk.CTkLabel(self.view_area, text="[ CAMERA OFF ]", font=("Arial", 20), text_color="#444")
        self.video_label.pack(expand=True, fill="both")

    def toggle(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened(): return
            self.is_running = True
            self.btn_main.configure(text="STOP SYSTEM", fg_color="#ff4d4d", hover_color="#cc0000")
            self.video_label.configure(text="")
            self.stream()
        else:
            self.is_running = False
            self.cap.release()
            self.btn_main.configure(text="START MONITORING", fg_color="#00ffcc")
            self.video_label.configure(image="", text="[ CAMERA OFF ]")
            self.progress.set(0)
            self.drowsy_acc = 0

    def siren(self):
        if not self.is_beeping:
            self.is_beeping = True
            for _ in range(3):
                winsound.Beep(2500, 200)
                winsound.Beep(3500, 200)
            self.is_beeping = False

    def stream(self):
        if self.is_running:
            ret, frame = self.cap.read()
            if ret:
                
                frame = cv2.flip(frame, 1)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                input_t = self.transform(img_pil).unsqueeze(0).to(self.DEVICE)

                
                with torch.no_grad():
                    output = self.model(input_t)
                    prob = torch.nn.functional.softmax(output, dim=1)
                    conf, pred = torch.max(prob, 1)

                state = self.CLASSES[pred.item()]
                confidence = conf.item()
                
                
                
                if state == 'Drowsy' and confidence > 0.65:
                    self.drowsy_acc += 0.15 
                else:
                    self.drowsy_acc -= 0.1  
                
            
                if self.drowsy_acc < 0: self.drowsy_acc = 0
                if self.drowsy_acc > 4.0: self.drowsy_acc = 4.0

           
                prog = min(self.drowsy_acc / 3.0, 1.0)
                self.progress.set(prog)
                
                if prog < 0.2:
                    self.progress.configure(progress_color="#00ffcc")
                    self.lbl_status.configure(text="🟢 DRIVER ALERT", text_color="#00ffcc")
                    color_border = (0, 255, 204)
                elif prog < 0.7:
                    self.progress.configure(progress_color="#ffcc00")
                    self.lbl_status.configure(text="🟡 WARNING", text_color="#ffcc00")
                    color_border = (0, 204, 255)
                else:
                    self.progress.configure(progress_color="#ff4d4d")
                    self.lbl_status.configure(text="🔴 WAKE UP!", text_color="#ff4d4d")
                    color_border = (0, 0, 255)
                    threading.Thread(target=self.siren, daemon=True).start()

                self.lbl_info.configure(text=f"Safety Level: {int((1-prog)*100)}%")

               
                cv2.rectangle(img_rgb, (0,0), (640, 480), color_border, 15)

          
                img_tk = ImageTk.PhotoImage(Image.fromarray(cv2.resize(img_rgb, (720, 480))))
                self.video_label.configure(image=img_tk)
                self.video_label._image = img_tk

            self.after(10, self.stream)

if __name__ == "__main__":
    app = VigilDriveFinal()
    app.mainloop()