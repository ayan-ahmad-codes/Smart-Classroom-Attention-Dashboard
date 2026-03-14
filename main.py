"""
main.py
-------
Entry point for the Smart Classroom Attention Detector.

Run with:
    python main.py

Controls:
    Q / ESC  → quit
    R        → reset blink counter
    S        → save screenshot
"""

import sys
import time

import cv2

from eye_tracker     import EyeTracker
from attention_logic import AttentionAnalyser
from utils           import draw_hud, CSVLogger


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration  (edit these if needed)
# ─────────────────────────────────────────────────────────────────────────────

CAMERA_INDEX  = 0       # 0 = default webcam; try 1 or 2 if that fails
FRAME_WIDTH   = 960
FRAME_HEIGHT  = 720
TARGET_FPS    = 30
LOG_FILE      = "attention_log.csv"
LOG_INTERVAL  = 5.0     # seconds between CSV rows
EAR_THRESHOLD = 0.22    # lower = blink triggers more easily
CONSEC_FRAMES = 3       # closed-eye frames needed to register a blink


# ─────────────────────────────────────────────────────────────────────────────

def open_camera(index: int) -> cv2.VideoCapture:
    """Open the webcam and set resolution / FPS."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera at index {index}.")
        print("        Try changing CAMERA_INDEX in main.py  (0, 1, 2 …)")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          TARGET_FPS)
    return cap


def main():
    print("=" * 52)
    print("   Smart Classroom Attention Detector")
    print("   Q / ESC → quit   R → reset blinks   S → screenshot")
    print("=" * 52)

    # ── Initialise all components ─────────────────────────────────────────────
    cap      = open_camera(CAMERA_INDEX)
    tracker  = EyeTracker(ear_threshold=EAR_THRESHOLD, consec_frames=CONSEC_FRAMES)
    analyser = AttentionAnalyser()
    logger   = CSVLogger(filepath=LOG_FILE, interval=LOG_INTERVAL)

    screenshot_count = 0
    prev_time        = time.time()

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame – retrying …")
            time.sleep(0.05)
            continue

        # Mirror the frame so it looks natural (like a webcam preview)
        frame = cv2.flip(frame, 1)

        # ── Eye / face tracking ───────────────────────────────────────────────
        has_face, ear, is_blink = tracker.process_frame(frame)

        # ── Attention scoring ─────────────────────────────────────────────────
        data = analyser.update(
            gaze_direction = tracker.gaze_direction,
            head_pose      = tracker.head_pose,
            ear            = ear,
            eyes_closed    = tracker.eyes_closed(),
            new_blink      = is_blink,
        )

        # Extra fields used by the HUD
        data["ear"]     = ear
        data["no_face"] = not has_face

        # ── Draw eye landmarks ────────────────────────────────────────────────
        if has_face:
            frame = tracker.draw_landmarks(frame)

        # ── Draw HUD overlay ──────────────────────────────────────────────────
        frame = draw_hud(frame, data)

        # ── FPS counter (top-right) ───────────────────────────────────────────
        curr_time = time.time()
        fps       = 1.0 / max(curr_time - prev_time, 1e-6)
        prev_time = curr_time
        h, w      = frame.shape[:2]
        cv2.putText(frame, f"FPS: {fps:.0f}",
                    (w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (180, 180, 180), 1, cv2.LINE_AA)

        # ── Display ───────────────────────────────────────────────────────────
        cv2.imshow("Attention Detector  [Q = quit]", frame)

        # ── Log to CSV ────────────────────────────────────────────────────────
        logger.log(data)

        # ── Keyboard controls ─────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):       # Q or ESC
            break
        elif key in (ord("r"), ord("R")):          # R – reset blink count
            tracker.reset_blink_count()
            print("[INFO] Blink counter reset.")
        elif key in (ord("s"), ord("S")):          # S – screenshot
            screenshot_count += 1
            fname = f"screenshot_{screenshot_count:03d}.png"
            cv2.imwrite(fname, frame)
            print(f"[INFO] Screenshot saved → {fname}")

    # ── Cleanup ───────────────────────────────────────────────────────────────
    tracker.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Session ended.  Log saved → {LOG_FILE}")


if __name__ == "__main__":
    main()