# AI-Based Threat Detection for Border Surveillance — Technical Documentation
**Version:** 1.0 · **Last updated:** 2025-08-29 20:07 UTC  
**Team:** Error 404: Sleep Not Found (Presidency University)

---

## 1. Overview
This document details the end-to-end system you built: a real-time border surveillance pipeline that detects **person** and **fence**, computes an **auto-anchored virtual fence line**, and classifies **PUSH‑IN** / **PUSH‑OUT** based on a tracked person’s **head point** relative to that line. The system sends alerts via **Twilio SMS/WhatsApp**, and runs in a loop against 20–30s clips or live streams.

**Core components**
- **Perception:** YOLO model (`best.pt`) trained on custom dataset (classes include *person*, *fence*).
- **Tracking:** ByteTrack via `ultralytics.YOLO().track(persist=True)` for stable IDs.
- **Rules Layer:** 
  - Auto-fence line from the largest fence bbox (smoothed with EMA).
  - Configurable **danger band** around the fence line (BAND_PX).
  - **Head-based** crossing detection (top-center of bbox).
  - Optional near-fence presence condition.
- **Alerting:** Twilio SMS/WhatsApp with per-ID cooldown; WhatsApp Sandbox supported.
- **Visualization:** Fixed-width preview window, annotated boxes, fence band, and push-in/out labels.
- **Looped Playback:** Replay the input clip seamlessly for demo and testing.

---

## 2. Objectives & Requirements
**Objectives**
- Early, reliable detection of near-fence incursions.
- Directional classification (**PUSH‑IN** vs **PUSH‑OUT**) for operational context.
- Low false positives; resilient to fence jitter and occlusions.
- Lightweight runtime suitable for edge (Jetson-class) or laptop demo.

**Functional Requirements**
- Ingest video file, RTSP, or webcam.
- Detect **person** & **fence** in real time.
- Maintain track IDs across frames.
- Anchor the decision boundary to detected fence (with smoothing & fallback).
- Send alerts on directional crossing or near-fence dwell (config switch).

**Non-Functional Requirements**
- Latency target: <1–2s from event to alert.
- Configurable thresholds (confidence, band size, cooldown).
- Graceful handling of dropped detections; no index errors.
- Secure handling of messaging credentials.

---

## 3. Dataset & Training
- **Classes**: `person`, `fence` (extendable to vehicles, animals, etc.).
- **Sources**: Your custom clips with varied lighting (day/night), weather, and camera angles.
- **Labeling**: Consistent bounding boxes for *fence* sections and *persons*; avoid fragmented fence labels where possible.
- **Augmentations**: Motion blur, brightness/contrast, night filters, random crops/scales.
- **Model Family**: YOLO (Ultralytics). Start with n/s/m variants based on device.
- **Hyperparameters (example)**:  
  `imgsz=640`, `epochs=100`, `batch=16`, `conf=0.35`, optimizer default.  
- **Validation**: Holdout clips; compute mAP(P) on *person*, per-class precision/recall, and real-world crossing accuracy.

---

## 4. Inference Pipeline
1. **Capture**: Load video (`SOURCE`) or camera index.
2. **Detect+Track**: `model.track(conf=CONF, persist=True, tracker="bytetrack.yaml")`.
3. **Auto-Fence**: From the largest `fence` bbox, pick the configured edge (`top|center|bottom`) → `fence_line_y`.
   - Smooth: `fence_y = alpha * candidate + (1-alpha) * fence_y` with `alpha = FENCE_SMOOTH_ALPHA`.
   - Hold if missing for `FENCE_HOLD_IF_MISSED` frames; fallback toward `FALLBACK_LINE_Y`.
4. **Danger Band (ROI)**: Create a horizontal band of thickness `BAND_PX` centered on `fence_line_y`.
5. **Head Point**: For each `person` track `tid`, compute `head = (center_x, y_top)` and clamp to frame.
6. **Crossing Logic**: Compare current `head.y` to previous `head.y` and the fence line:
   - `prev < fence_y and curr >= fence_y` → **PUSH‑IN** (top→bottom).
   - `prev > fence_y and curr <= fence_y` → **PUSH‑OUT** (bottom→top).
7. **Trigger Policy**: 
   - If `REQUIRE_CROSSING=True`: alert only on directional crossing.
   - Else: also alert when head stays inside band for `MIN_STAY_FRAMES`.
8. **Throttle**: Per `tid` cooldown `ALERT_COOLDOWN_S`.
9. **Alert**: Twilio `messages.create(...)`; on-screen label persists `ONSCREEN_DIR_FRAMES` frames.

---

## 5. Configuration Keys (code constants)
```python
# Detection
CONF = 0.35

# Auto-fence
USE_AUTO_FENCE = True
FENCE_EDGE = "bottom"          # or "center", "top"
FENCE_SMOOTH_ALPHA = 0.20
FENCE_HOLD_IF_MISSED = 25
FALLBACK_LINE_Y = 235
LINE_OFFSET_PX = 0             # negative raises line; positive lowers

# Band
BAND_PX = 60                   # increase to widen near-fence area

# Policy
REQUIRE_CROSSING = True
MIN_STAY_FRAMES = 6
ALERT_COOLDOWN_S = 45
ONSCREEN_DIR_FRAMES = 35

# Orientation labels
SIDE_TOP = "India"
SIDE_BOTTOM = "Bangladesh"

# IO
SOURCE = "input.mp4"           # or 0 for webcam
DISPLAY_W = 1280               # preview width
```

---

## 6. Alerting & Messaging (Twilio)
**Channels**: SMS (long code, Toll-Free), **WhatsApp Sandbox** (trial-friendly).  
**Trial rules**: You must verify recipient numbers; WhatsApp Sandbox needs a one-time "join" from the recipient.  
**Production (country-specific)**: Use local senders or registered Sender IDs for best deliverability; US SMS needs A2P 10DLC or verified Toll-Free.

**Code hook**
```python
from twilio.base.exceptions import TwilioRestException

def send_alert(text):
    try:
        client.messages.create(body=text, from_=FROM_NUMBER, to=TO_NUMBER)
        print("[TWILIO] Sent:", text)
    except TwilioRestException as e:
        print(f"[TWILIO ERROR] status={e.status} code={e.code} msg={e.msg}")
```

**Security**: Store `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, and numbers as environment variables or in a secrets store, not in code.

---

## 7. Deployment
**Laptop Demo**: Python 3.10+, `ultralytics`, `opencv-python`, `twilio`.  
**Edge Device (Jetson)**: Use FP16/INT8 exports; pin compatible CUDA/cuDNN; enable `--half` if supported.  
**Containers**: Package inference + alert service; mount config; use RTSP/RTMP inputs.  
**Looped Playback**: 20–30s demo clip; replay automatically for demos.

---

## 8. Logging & Metrics
- Console logs: model FPS, detection counts, Twilio status/error codes.
- Event log (optional): JSON lines with `{timestamp, tid, direction, conf, frame_no}`.
- KPIs: precision/recall on *person*, crossing detection accuracy, false alert rate, alert latency.

---

## 9. Testing & Validation
- **Unit**: crossing logic on synthetic sequences (head.y above/below fence).
- **Integration**: full pipeline on three clips (day/night/occluded).
- **Field**: short pilot in a controlled sector with human-in-the-loop verification.
- Record edge cases: multiple persons near fence, fence occluded, camera tilt.

---

## 10. Troubleshooting
- **No window appears**: ensure `SHOW_WINDOW=True` and a non-headless session.
- **IndexError on ROI**: clamp `cx, cy` to `[0..w-1], [0..h-1]` (already implemented).
- **Fence jitter**: raise `FENCE_SMOOTH_ALPHA` slightly or increase `FENCE_HOLD_IF_MISSED`.
- **Too many alerts**: increase `ALERT_COOLDOWN_S`, widen thresholds, or set `REQUIRE_CROSSING=True`.
- **Twilio blocked**: for trials, verify recipient; for US SMS, use A2P 10DLC or verified Toll-Free; for international, prefer local senders.

---

## 11. Roadmap
- Thermal fusion for night ops; depth estimation for distance-to-fence in meters.
- Multi-zone policy (warning vs red zone) with different actions.
- Upload event thumbnails to storage; link in alert.
- Ops dashboard with map and evidence review.
- Federated learning loop for continuous improvement.

---

## 12. Appendix — Example Env Vars
```
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_FROM=whatsapp:+14155238886
TWILIO_TO=whatsapp:+8801XXXXXXXXX
```
