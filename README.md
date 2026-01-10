# Falcon AI — AI-Based Threat Detection for Border Surveillance

**Falcon AI** detects **person** and **fence** in real time, builds a **virtual fence line**, and classifies **PUSH-IN** vs **PUSH-OUT** using the **head point** of each tracked person. It draws a near-fence “danger band” for early warnings and can send alerts via **Twilio SMS** or **WhatsApp**.  
This project currently demonstrates ground-camera videos (we don’t have a drone yet). Border data is sensitive, so we trained on a **custom dataset we filmed on our phones**.

---

## ✨ Key Features
- **Custom YOLO model (`best.pt`)** trained on *person* & *fence*  
- **Auto-anchored virtual fence** from detected fence (with smoothing + fallback)  
- **Head-based direction logic**
  - head **above → below** line ⇒ **PUSH-IN**  
  - head **below → above** line ⇒ **PUSH-OUT**
- **Near-fence band** (configurable width) for presence warnings  
- **Twilio alerts** (SMS/WhatsApp) with **per-ID cooldown** (no spam)  
- **Looped 20–30s demo** playback + on-screen overlays and counters  

---

## 🗂 Suggested Repo Layout
```
project/
├─ New_Model/
│  ├─ MODEL_FILE/best.pt
│  └─ IN_OUT_VIDEO/14.mp4
├─ Border_threat_detection.py          # main detection + alerting
├─ scripts/                            # (optional) intro/outro/demo generators
├─ docs/
│  ├─ Architecture_Diagram.svg
│  ├─ Data_Pipeline_Flow.svg
│  └─ Border_Threat_Detection_Technical_Documentation.md
└─ README.md
```
> If you already use different paths, just update the constants in your script.

---

## 🧰 Requirements
- **Python 3.9–3.11** (tested on Windows)
- Packages: `ultralytics`, `opencv-python`, `numpy`, `twilio`  
  *(Optional tools: `moviepy`, `pydub`, `pyttsx3`, `edge-tts`, `streamlit`)*

**Setup**
```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # macOS/Linux
pip install -U ultralytics opencv-python numpy twilio
```

---

## ⚙️ Configuration (copy into your script’s CONFIG block)
```python
# Paths
MODEL_PATH   = r"D:\Shihab_files\AI_Based_Threat_Detection_for_Border_Surveillance\New_Model\MODEL_FILE\best.pt"
SOURCE       = r"D:\Shihab_files\AI_Based_Threat_Detection_for_Border_Surveillance\New_Model\IN_OUT_VIDEO\14.mp4"
CONF         = 0.35            # detection confidence threshold

# Auto-fence (preferred)
USE_AUTO_FENCE       = True
FENCE_EDGE           = "bottom"   # "top" | "center" | "bottom" -> which fence edge becomes the line
FENCE_SMOOTH_ALPHA   = 0.20       # 0..1 (higher = quicker line updates)
FENCE_HOLD_IF_MISSED = 25         # keep last line when fence not visible (frames)
FALLBACK_LINE_Y      = 300        # used until fence is seen
LINE_OFFSET_PX       = 0          # negative = line up, positive = down

# Near-fence “danger” band (pixels)
BAND_PX              = 70         # increase to widen the zone

# Policy
REQUIRE_CROSSING     = True       # True: alert only on push-in/out; False: also alert on dwell in band
MIN_STAY_FRAMES      = 6          # used if REQUIRE_CROSSING=False
ALERT_COOLDOWN_S     = 45         # per track-id
ONSCREEN_DIR_FRAMES  = 35         # how long to keep the direction banner (frames)

# Orientation labels (shown on screen)
SIDE_TOP             = "India"
SIDE_BOTTOM          = "Bangladesh"

# Window / render
SHOW_WINDOW          = True
DISPLAY_W            = 1280       # preview width

# Twilio (recommended: set as environment variables)
# TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM, TWILIO_TO
```

**Quick tweaks**
- Move the line **up**: `LINE_OFFSET_PX = -20` (or more negative)  
- Make the zone **thicker**: `BAND_PX = 120`  

---

## ▶️ Run
```bash
.venv\Scripts\activate
python Border_threat_detection.py
```
- Press **Q** to close the live window.  
- The script plays the input clip end-to-end; enable looping in your code if needed.

---

## 🔔 Alerts (Twilio SMS / WhatsApp)

**Set environment variables**

**Windows (CMD)**
```cmd
set TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
set TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
set TWILIO_FROM=whatsapp:+14155238886   # or a phone number: +1..., +44..., etc.
set TWILIO_TO=whatsapp:+8801XXXXXXXXX   # or a phone number: +61..., +44..., +1...
```

**PowerShell**
```powershell
$env:TWILIO_ACCOUNT_SID="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
$env:TWILIO_AUTH_TOKEN="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
$env:TWILIO_FROM="whatsapp:+14155238886"
$env:TWILIO_TO="whatsapp:+8801XXXXXXXXX"
```

**macOS/Linux**
```bash
export TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
export TWILIO_FROM=whatsapp:+14155238886
export TWILIO_TO=whatsapp:+8801XXXXXXXXX
```

**Notes**
- **Trial** accounts must **verify recipient numbers**; or use the **WhatsApp Sandbox**.  
- For **US SMS**, production traffic typically needs **A2P 10DLC** or **verified toll-free**.  
- For non-US recipients, local senders usually deliver best (e.g., UK number for UK).

---

## 🧠 How Falcon AI Decides Direction
1. Detect **person** & **fence** per frame  
2. Build/smooth the **virtual fence line** from the largest fence box  
3. Track each person (stable ID)  
4. Use the **head point** (top-center of the person box) vs the line:
   - head **above → below** ⇒ **PUSH-IN**
   - head **below → above** ⇒ **PUSH-OUT**
5. If `REQUIRE_CROSSING=False`, also alert when head **dwells** inside the blue **BAND_PX** zone  
6. Throttle alerts per ID using `ALERT_COOLDOWN_S`

---

## 🧪 Demo / Prototype Videos (optional)
We include small **OpenCV** scripts (in `scripts/`) to generate:
- 5s title cards (Twilio/WhatsApp, LIVE push-in/out)  
- 12s intro  
- Full **prototype demo** with overlays and HUD  

Merge clips with **MoviePy** or **ffmpeg** as you like.

---

## 🧩 Troubleshooting
- **No window / no video** → `SHOW_WINDOW=True`, confirm `SOURCE` path, ensure a display  
- **IndexError (ROI/coords)** → coordinate clamping is included; pull latest script  
- **Fence not detected** → lower `CONF` (e.g., 0.30), ensure fence labeled as a single large bbox, check lighting  
- **Line too jittery** → lower `FENCE_SMOOTH_ALPHA` (e.g., 0.15) or raise `FENCE_HOLD_IF_MISSED`  
- **Too many alerts** → increase `ALERT_COOLDOWN_S`, set `REQUIRE_CROSSING=True`, widen thresholds  
- **Twilio errors** → verify recipient for trials; use Sandbox for WhatsApp; use proper local senders in production

---

## 🔒 Ethics & Safety
- **Purpose:** reduce confusion and **prevent harm** near borders by providing earlier, clearer context  
- **Data:** border footage is **sensitive**; we used our **own phone videos** for training and will only use approved data in future  
- **Human-in-the-loop:** alerts support trained officers—they do **not** replace them

---

## 🗺 Roadmap
- Add **thermal/IR** for low-light and fog  
- Export to **FP16/INT8** (ONNX/TensorRT) for Jetson-class devices  
- Multi-zone policies (warning vs red zones)  
- Evidence snapshots to secure storage (link in alert)  
- Ops dashboard + MLOps; explore **federated learning**

---

## 🙌 Team
**Error 404: Sleep Not Found** — Presidency University  
Roles: AI/edge, CV pipeline, systems, research, and ops

**System name:** **Falcon AI**

---

## 📄 License & Contact
- License: *(add your license here)*  
- Contact: *(email/phone/link here)*

> For deeper detail, include `docs/Architecture_Diagram.svg`, `docs/Data_Pipeline_Flow.svg`, and `docs/Border_Threat_Detection_Technical_Documentation.md` in your repo.
