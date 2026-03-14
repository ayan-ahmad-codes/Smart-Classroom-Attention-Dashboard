"""
utils.py
--------
Utility functions used by main.py:
  - draw_hud()     : draws the on-screen info overlay
  - CSVLogger      : writes attention data to a CSV file
"""

import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np


# ──────────────────────────────────────────────
#  Colour palette  (BGR)
# ──────────────────────────────────────────────
GREEN  = (50, 205, 50)
YELLOW = (0, 200, 255)
RED    = (50, 50, 220)
WHITE  = (255, 255, 255)
BLACK  = (0, 0, 0)
CYAN   = (255, 200, 0)
ORANGE = (0, 140, 255)


def status_color(status: str) -> tuple:
    """Return a colour for each attention status."""
    return {
        "Attentive":  GREEN,
        "Distracted": YELLOW,
        "Sleepy":     RED,
    }.get(status, WHITE)


def score_color(score: int) -> tuple:
    """Gradient: green (high score) → yellow → red (low score)."""
    if score >= 70:
        return GREEN
    elif score >= 45:
        return YELLOW
    else:
        return RED


# ──────────────────────────────────────────────
#  HUD (Heads-Up Display) overlay
# ──────────────────────────────────────────────

def draw_hud(frame: np.ndarray, data: dict) -> np.ndarray:
    """
    Draws a semi-transparent info panel on the webcam feed.

    Parameters
    ----------
    frame : BGR numpy array from OpenCV
    data  : dict returned by AttentionAnalyser.update()

    Returns modified frame.
    """
    h, w = frame.shape[:2]

    score  = data.get("attention_score", 0)
    status = data.get("status", "---")
    gaze   = data.get("gaze_direction", "---")
    head   = data.get("head_pose", "---")
    blinks = data.get("blink_rate", 0.0)
    alert  = data.get("alert", False)
    ear    = data.get("ear", 0.0)

    # ── Semi-transparent panel background ───────────────
    panel_w = 280
    panel_h = 200
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), BLACK, -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # ── Attention Score ──────────────────────────────────
    cv2.putText(frame, "Attention Score",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{score}%",
                (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.6, score_color(score), 2, cv2.LINE_AA)

    # ── Score bar ────────────────────────────────────────
    bar_x, bar_y, bar_h = 20, 85, 12
    bar_max_w = panel_w - 20
    filled = int(bar_max_w * score / 100)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_max_w, bar_y + bar_h), (60, 60, 60), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), score_color(score), -1)

    # ── Status label ────────────────────────────────────
    cv2.putText(frame, f"Status: {status}",
                (20, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color(status), 1, cv2.LINE_AA)

    # ── Detail lines ────────────────────────────────────
    cv2.putText(frame, f"Gaze: {gaze}   Head: {head}",
                (20, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Blink rate: {blinks:.1f} /min   EAR: {ear:.2f}",
                (20, 162), cv2.FONT_HERSHEY_SIMPLEX, 0.45, CYAN, 1, cv2.LINE_AA)

    # ── Clock ────────────────────────────────────────────
    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, ts, (20, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    # ── ALERT banner ─────────────────────────────────────
    if alert:
        _draw_alert(frame, w, h)

    # ── No-face indicator ────────────────────────────────
    if data.get("no_face"):
        cv2.putText(frame, "No face detected",
                    (w // 2 - 120, h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 2, cv2.LINE_AA)

    return frame


def _draw_alert(frame: np.ndarray, w: int, h: int):
    """Flash a warning banner at the bottom of the frame."""
    # Pulsing effect based on time
    pulse = int(abs(np.sin(time.time() * 3)) * 255)
    color = (0, 0, pulse)   # red with intensity pulse

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 70), (w, h), color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame,
                "⚠  Please focus on the lecture!",
                (w // 2 - 220, h - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)


# ──────────────────────────────────────────────
#  CSV Logger
# ──────────────────────────────────────────────

class CSVLogger:
    """
    Appends attention data rows to a CSV file every LOG_INTERVAL seconds.

    CSV columns: timestamp, attention_score, status, gaze, head_pose, blink_rate
    """

    def __init__(self, filepath: str = "attention_log.csv", interval: float = 5.0):
        self.filepath     = filepath
        self.interval     = interval
        self._last_log    = time.time()

        # Create the file with a header row if it doesn't exist
        if not os.path.isfile(filepath):
            with open(filepath, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "attention_score", "status",
                    "gaze_direction", "head_pose", "blink_rate_per_min"
                ])

    def log(self, data: dict):
        """Write a row if the interval has elapsed."""
        now = time.time()
        if now - self._last_log < self.interval:
            return

        self._last_log = now
        row = [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data.get("attention_score", 0),
            data.get("status", ""),
            data.get("gaze_direction", ""),
            data.get("head_pose", ""),
            data.get("blink_rate", 0.0),
        ]
        with open(self.filepath, "a", newline="") as f:
            csv.writer(f).writerow(row)
