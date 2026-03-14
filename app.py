"""
app.py
------
Flask-powered web dashboard for the Smart Classroom Attention Detector.

Run with:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import csv
import os
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template

from eye_tracker import EyeTracker
from attention_logic import AttentionAnalyser
from utils import draw_hud, CSVLogger


# ─────────────────────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────────────────────

CAMERA_INDEX  = 0
FRAME_WIDTH   = 960
FRAME_HEIGHT  = 720
TARGET_FPS    = 30
LOG_FILE      = "attention_log.csv"
LOG_INTERVAL  = 5.0
EAR_THRESHOLD = 0.22
CONSEC_FRAMES = 3

# ─────────────────────────────────────────────────────────────────────────────
#  Flask App
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)

# Shared state (written by the capture thread, read by the API)
_lock = threading.Lock()
_latest_data: dict = {}
_latest_jpeg: bytes = b""


# ─────────────────────────────────────────────────────────────────────────────
#  Background capture thread
# ─────────────────────────────────────────────────────────────────────────────

def _capture_loop():
    """Run camera capture + processing in a background thread."""
    global _latest_data, _latest_jpeg

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check CAMERA_INDEX in app.py.")
        return

    tracker  = EyeTracker(ear_threshold=EAR_THRESHOLD, consec_frames=CONSEC_FRAMES)
    analyser = AttentionAnalyser()
    logger   = CSVLogger(filepath=LOG_FILE, interval=LOG_INTERVAL)

    print("[INFO] Camera capture thread started.")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        frame = cv2.flip(frame, 1)

        # ── Eye / face tracking ──────────────────────────────────────────
        has_face, ear, is_blink = tracker.process_frame(frame)

        # ── Attention scoring ────────────────────────────────────────────
        data = analyser.update(
            gaze_direction=tracker.gaze_direction,
            head_pose=tracker.head_pose,
            ear=ear,
            eyes_closed=tracker.eyes_closed(),
            new_blink=is_blink,
            has_face=has_face,
        )
        data["ear"]         = int(ear * 1000) / 1000.0
        data["no_face"]     = not has_face
        data["blink_count"] = tracker.get_blink_count()

        # ── Draw landmarks + HUD ─────────────────────────────────────────
        if has_face:
            frame = tracker.draw_landmarks(frame)
        frame = draw_hud(frame, data)

        # ── Encode JPEG ──────────────────────────────────────────────────
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

        with _lock:
            _latest_data = data
            _latest_jpeg = jpeg.tobytes()

        # ── Log to CSV ───────────────────────────────────────────────────
        logger.log(data)

        time.sleep(1.0 / TARGET_FPS)

    cap.release()
    tracker.close()


# Start capture thread on import
_thread = threading.Thread(target=_capture_loop, daemon=True)
_thread.start()


# ─────────────────────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG stream of the annotated camera feed."""
    def generate():
        while True:
            with _lock:
                frame = _latest_jpeg
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                )
            time.sleep(1.0 / TARGET_FPS)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/data")
def api_data():
    """Return the latest attention metrics as JSON."""
    with _lock:
        data = dict(_latest_data)
    return jsonify(data)


@app.route("/api/history")
def api_history():
    """Return the last 100 rows from the CSV log."""
    rows = []
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), LOG_FILE)
    if os.path.isfile(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    # Return last 100 entries (newest first)
    last_100 = rows[-100:]
    last_100.reverse()
    return jsonify(last_100)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 52)
    print("   Smart Classroom Attention Detector — Dashboard")
    print("   Open http://127.0.0.1:5000 in your browser")
    print("=" * 52)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
