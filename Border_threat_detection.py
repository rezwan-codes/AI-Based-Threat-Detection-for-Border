import os, time, cv2, numpy as np
from ultralytics import YOLO
from twilio.rest import Client

# ------------- CONFIG -------------
MODEL_PATH   = "D://Shihab_files//AI_Based_Threat_Detection_for_Border_Surveillance//New_Model//MODEL_FILE//best.pt"
SOURCE       = "D://Shihab_files//AI_Based_Threat_Detection_for_Border_Surveillance//New_Model//IN_OUT_VIDEO//9.mp4"
CONF         = 0.35  # detection confidence

# Fence handling (auto from detected fence, smoothed)
USE_AUTO_FENCE       = True
FENCE_EDGE           = "bottom"   # "bottom" | "center" | "top"
FENCE_SMOOTH_ALPHA   = 0.20
FENCE_HOLD_IF_MISSED = 25

# Fallback/initial line and band
LINE_Y            = 215
FALLBACK_LINE_Y   = LINE_Y
BAND_PX           = 500
LINE_OFFSET_PX = -20

# Loop & display
LOOP_FOREVER = True
NUM_LOOPS    = 0
SHOW_WINDOW  = True
DISPLAY_W    = 1280
PAUSE_BETWEEN_LOOPS_S = 0.5

# Alert policy
REQUIRE_CROSSING = True          # True = only push-in/out; False = also near-fence presence
ALERT_COOLDOWN_S = 45
MIN_STAY_FRAMES  = 6             # used only when REQUIRE_CROSSING=False

# Side labels (edit to match your sector)
SIDE_TOP         = "India"
SIDE_BOTTOM      = "Bangladesh"
PUSHIN_LABEL     = f"{SIDE_TOP} ➜ {SIDE_BOTTOM} (push-in)"
PUSHOUT_LABEL    = f"{SIDE_BOTTOM} ➜ {SIDE_TOP} (push-out)"

# On-screen direction label lifetime
ONSCREEN_DIR_FRAMES = 35

# Twilio (use env vars in production)
TWILIO_SID   = os.getenv("TWILIO_ACCOUNT_SID", "sid")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN",  "AUTH_TOKEN")
FROM_NUMBER  = os.getenv("TWILIO_FROM",        "whatsapp:+14155238")     # or 'whatsapp:+14155238886'
TO_NUMBER    = os.getenv("TWILIO_TO",          "whatsapp:+880190")   # or 'whatsapp:+8801XXXXXXX'
# ----------------------------------

client = Client(TWILIO_SID, TWILIO_TOKEN)
model = YOLO(MODEL_PATH)

# Global alert state across loops
last_alert_ts = {}            # track_id -> last alert time
show_dir_until = {}           # track_id -> frame index until which to display direction

# ---------- helpers ----------
def send_alert(text):
    try:
        client.messages.create(body=text, from_=FROM_NUMBER, to=TO_NUMBER)
        print("[TWILIO] Sent:", text)
    except Exception as e:
        print("[TWILIO ERROR]", e)

def scale_for_display(img, target_w):
    h, w = img.shape[:2]
    if w == target_w: return img
    s = target_w / float(w)
    return cv2.resize(img, (target_w, int(h*s)), interpolation=cv2.INTER_LINEAR)

def clamp(v, lo, hi):
    return int(max(lo, min(hi, v)))

def build_roi(h, w, line_y, band_px):
    y1 = max(0, int(line_y) - band_px // 2)
    y2 = min(h - 1, int(line_y) + band_px // 2)
    poly = np.array([[0,y1],[w,y1],[w,y2],[0,y2]], np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return poly, mask

def pick_fence_line_from_boxes(boxes, names, h):
    """Pick candidate line from largest fence bbox; return y or None."""
    if boxes is None or len(boxes) == 0: return None
    xyxy = boxes.xyxy.cpu().numpy()
    clss = boxes.cls.cpu().numpy()
    best_area, best_y = -1, None
    for (x1,y1,x2,y2), c in zip(xyxy, clss):
        if names[int(c)] != "fence": continue
        area = max(1.0, (x2 - x1) * (y2 - y1))
        if area > best_area:
            best_area = area
            if FENCE_EDGE == "top":       best_y = int(y1)
            elif FENCE_EDGE == "center":  best_y = int((y1 + y2) / 2)
            else:                          best_y = int(y2)  # bottom
    if best_y is None: return None
    return int(np.clip(best_y, 0, h-1))
# -----------------------------

def run_one_pass():
    names = None
    inside_counter = {}   # id -> consecutive frames inside band (used when REQUIRE_CROSSING=False)
    last_y = {}           # id -> previous HEAD y (top of bbox)
    fence_line_y = FALLBACK_LINE_Y
    last_fence_seen_frame = -10**9
    frame_idx = 0
    win_name = "Border Threat Monitor"
    first_frame = True

    for result in model.track(
        source=SOURCE, conf=CONF, stream=True, persist=True, tracker="bytetrack.yaml"
    ):
        frame = result.orig_img.copy()
        h, w = frame.shape[:2]

        if SHOW_WINDOW and first_frame:
            try: cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            except: pass
            first_frame = False

        if names is None:
            names = result.names

        # ---- auto-fence from detection (smoothed) ----
        if USE_AUTO_FENCE:
            cand = pick_fence_line_from_boxes(result.boxes, names, h)
            if cand is not None:
                fence_line_y = int(FENCE_SMOOTH_ALPHA * cand + (1 - FENCE_SMOOTH_ALPHA) * fence_line_y)
                last_fence_seen_frame = frame_idx
            elif frame_idx - last_fence_seen_frame > FENCE_HOLD_IF_MISSED:
                fence_line_y = int(0.5 * fence_line_y + 0.5 * FALLBACK_LINE_Y)

        # ROI for this frame
        roi_poly, roi_mask = build_roi(h, w, fence_line_y, BAND_PX)

        # Draw fence band & line
        cv2.polylines(frame, [roi_poly], True, (255,0,0), 2)
        cv2.line(frame, (0, fence_line_y), (w, fence_line_y), (255,0,0), 2)
        cv2.putText(frame, f"FENCE ZONE ({'AUTO' if USE_AUTO_FENCE else 'FIXED'})",
                    (10, max(28, fence_line_y - BAND_PX//2 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        # ---- detections ----
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            ids  = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.array([-1]*len(xyxy))

            for bbox, c, tid in zip(xyxy, clss, ids):
                cls = names[int(c)]

                # Draw fence boxes (for visibility)
                if cls == "fence":
                    fx1, fy1, fx2, fy2 = map(int, bbox)
                    fx1 = clamp(fx1, 0, w-1); fx2 = clamp(fx2, 0, w-1)
                    fy1 = clamp(fy1, 0, h-1); fy2 = clamp(fy2, 0, h-1)
                    cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 255), 2)
                    cv2.putText(frame, "fence", (fx1, max(20, fy1-8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
                    continue

                if cls != "person" or tid == -1:
                    continue

                x1,y1b,x2,y2b = map(int, bbox)
                # clamp bbox
                x1 = clamp(x1, 0, w-1); x2 = clamp(x2, 0, w-1)
                y1b= clamp(y1b,0, h-1); y2b= clamp(y2b,0, h-1)

                # HEAD point (top-center)
                cx = clamp(int((x1 + x2) / 2), 0, w-1)
                cy = clamp(int(y1b), 0, h-1)

                # draw person and head point
                cv2.rectangle(frame, (x1,y1b), (x2,y2b), (0,255,255), 2)
                cv2.circle(frame, (cx,cy), 5, (0,255,255), -1)
                cv2.putText(frame, f"id:{tid}", (x1, max(20, y1b-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

                # crossing detection (using HEAD Y vs fence line)
                prev_y = last_y.get(tid, cy)
                last_y[tid] = cy
                crossed_down = (prev_y < fence_line_y and cy >= fence_line_y)  # head moved above -> below
                crossed_up   = (prev_y > fence_line_y and cy <= fence_line_y)  # head moved below -> above
                direction = PUSHIN_LABEL if crossed_down else (PUSHOUT_LABEL if crossed_up else None)

                # near-fence presence counter (only used if not requiring crossing)
                inside = roi_mask[cy, cx] > 0
                inside_counter[tid] = (inside_counter.get(tid, 0) + 1) if inside else 0

                # choose trigger condition
                trigger = (direction is not None) if REQUIRE_CROSSING else (inside_counter[tid] >= MIN_STAY_FRAMES)

                # throttled alert
                now = time.time()
                if trigger and (now - last_alert_ts.get(tid, 0)) > ALERT_COOLDOWN_S:
                    label = direction if direction is not None else "Near-fence presence"
                    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    msg = f"THREAT ALERT: Person #{tid} {label} @ {stamp}"
                    send_alert(msg)
                    last_alert_ts[tid] = now
                    show_dir_until[tid] = frame_idx + ONSCREEN_DIR_FRAMES

                # persistent on-screen direction label
                if show_dir_until.get(tid, 0) > frame_idx:
                    text = direction if direction is not None else "NEAR FENCE"
                    color = (0,0,255) if ("push-in" in text) else (0,255,0)
                    cv2.putText(frame, text, (x1, max(20, y1b - 24)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # show window
        if SHOW_WINDOW:
            disp = scale_for_display(frame, DISPLAY_W)
            cv2.imshow(win_name, disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return True

        frame_idx += 1

    return False  # pass completed

def main():
    loops_done = 0
    while True:
        quit_req = run_one_pass()
        if quit_req: break
        loops_done += 1
        if not LOOP_FOREVER and loops_done >= NUM_LOOPS: break
        time.sleep(PAUSE_BETWEEN_LOOPS_S)
    try: cv2.destroyAllWindows()
    except: pass
    print("Stopped.")

if __name__ == "__main__":
    main()
