# VigilDrive

<div align="center">

### A driver-monitoring system that detects drowsiness and helps reduce fatigue-related crashes

[Հայերեն](README.md) · [Русский](README.ru.md) · [English](README.en.md)

</div>

VigilDrive is a Windows driver-monitoring system. It uses a camera and a custom machine-learning model to detect drowsiness. If the driver’s eyes remain closed too long, it plays an audible alarm and sends subscribed relatives a photo, warning, and vehicle location through Telegram.

> **Safety notice.** VigilDrive is an assistive system. It does not replace driver attention, rest, or responsibility and does not guarantee accident prevention. Never drive while drowsy.

## Features

- Real-time face and eye monitoring;
- drowsiness classification with `vigil_model_v7_best`;
- eye-closure duration tracking;
- audible alarm after 2 seconds;
- emergency notification after 15 seconds;
- Telegram delivery of a photo, warning, and location;
- relative subscriptions using a driver code;
- browser GPS with IP-based fallback;
- automatic CUDA/CPU selection.

## How it works

The application reads camera frames, classifies drowsiness with EfficientNet-B0, and measures eye openness using MediaPipe Face Mesh and EAR. After 2 seconds of continuous eye closure it plays an alarm. After 15 seconds it captures the current frame and sends an alert to the backend. Telegram subscribers receive the image, warning, map link or coordinates, and timestamp.

## Technology

Python, PyTorch, torchvision, EfficientNet-B0, OpenCV, MediaPipe Face Mesh, CustomTkinter, FastAPI, Uvicorn, SQLite, Telegram Bot API, Browser Geolocation API, OpenStreetMap Nominatim, and IP geolocation fallback.

## Windows usage

The packaged executable is called **Sunlight**. Windows 11 is the primary target; Windows 10 may work but is not officially confirmed. Run the executable, register with the Telegram bot using `/register`, enter the returned driver code, give it to trusted relatives for `/subscribe CODE`, and press **START MONITORING**.

## Run from source

```powershell
git clone https://github.com/27Narek27/VigilDrive.git
cd VigilDrive
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place `vigil_model_v7_best.pth` in the project root and run in separate terminals:

```powershell
uvicorn Server:app --host 127.0.0.1 --port 8000
python bot.py
python main.py
```

Keep the Telegram bot token in a secret environment variable or secret manager. Never commit it to source control.

## Location

The app first requests high-accuracy browser GPS through a temporary localhost page. If GPS is unavailable, it falls back to approximate IP geolocation. GPS is reverse-geocoded with OpenStreetMap Nominatim; IP data may identify only a city or region.

## Model and training

`vigil_model_v7_best.pth` is an EfficientNet-B0 checkpoint with `Drowsy` and `Non Drowsy` classes. [train.py](train.py) expects an ImageFolder dataset under `final_dataset/` with `Drowsy/` and `Non Drowsy/` directories. Training uses 20 epochs, a 15% validation split, and batch size 32.

## Limitations

Windows 11 is the primary target. Poor lighting, glasses, camera position, and unavailable GPS may reduce reliability. Telegram alerts require network access. The system does not replace human supervision or rest.

## Contact and license

- **Narek Hakobyan** — [GitHub](https://github.com/27Narek27), [Telegram](https://t.me/Narekhakobyan133), [email](mailto:hakobyannarek204@gmail.com)
- **Robert Safaryan**

VigilDrive is distributed under the [MIT License](LICENSE). Copyright © 2025 Narek Hakobyan.

## Configuration and project status

This is a Windows demonstration prototype for drivers, including truck drivers. The team's trained model uses EfficientNet-B0 initialized with torchvision pretrained weights. It is not a newly designed neural architecture. Published accuracy metrics and road validation are not included.

After installing dependencies, run `Copy-Item .env.example .env` only if no local `.env` exists. Set `TELEGRAM_BOT_TOKEN` to your private bot token. Both server and bot load it automatically; existing environment variables take priority. Run only one polling process per bot.

Use the virtual environment in every terminal. The server defaults to local port 8000; public deployment requires separate authentication and HTTPS hardening.

Model weights and Sunlight are not included in the current source checkout. Contact the authors for available builds and weights; no verified direct download is provided. Source execution needs Python compatible with PyTorch and MediaPipe wheels. MediaPipe 0.10.21 is selected for the legacy Face Mesh API; the full dependency environment has not been integration-tested.

Local inference and sound work offline once the model is installed. Telegram, IP location and reverse geocoding use the network. Browser geolocation may rely on Wi-Fi or other sources rather than GPS hardware. Coordinates are cached when monitoring starts, not continuously updated while driving.

EAR and model predictions jointly influence detection. The 2/15-second thresholds use the internal closure timer; 1.2 seconds of open eyes resets it. Photos and location are transmitted to the server and Telegram. SQLite stores driver names, chat IDs and subscriptions. Share driver codes only with trusted contacts.

## Checks and contributions

Run `python -m unittest discover -s tests -v` to test Telegram response handling without sending real messages. Test camera, audio and notifications at a desk or in a stationary vehicle. Submit issues and focused pull requests with verification steps. These tests do not establish road safety.
