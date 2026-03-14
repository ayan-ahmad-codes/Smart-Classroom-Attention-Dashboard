"""
attention_logic.py
------------------
Contains the AttentionAnalyser class that:
  - Combines gaze, blink rate, and head pose into an Attention Score (0-100)
  - Determines the attention status label (Attentive / Distracted / Sleepy)
  - Manages a rolling time window for blink-rate calculations
  - Tracks how long the score has been below the alert threshold
"""

import time
from collections import deque



# ──────────────────────────────────────────────
#  Tunable constants
# ──────────────────────────────────────────────

GAZE_WEIGHT       = 0.70   # 70 % of score
BLINK_WEIGHT      = 0.20   # 20 %
HEAD_WEIGHT       = 0.10   # 10 %

NORMAL_BLINK_RATE = 15     # blinks per minute (healthy baseline)
MAX_BLINK_RATE    = 30     # above this is considered very high (sleepiness)

SLEEP_EAR_FRAMES  = 60     # ~2 seconds at 30 fps → classified as "Sleepy"
ALERT_THRESHOLD   = 40     # attention score below this triggers a warning
ALERT_DURATION    = 5.0    # seconds the score must stay below threshold before alert fires

WINDOW_SECONDS    = 60     # rolling window for blink-rate calculation


class AttentionAnalyser:
    """
    Calculates a 0-100 Attention Score every frame.

    Formula
    -------
    score = (gaze_score * GAZE_WEIGHT
           + blink_score * BLINK_WEIGHT
           + head_score  * HEAD_WEIGHT) * 100
    """

    def __init__(self):
        # Timestamps of recent blinks used for rolling blink-rate
        self._blink_times: deque = deque()

        # Tracks consecutive frames where eyes are closed (for sleepiness)
        self._closed_frame_count = 0

        # Time tracking for the alert system (-1.0 means "not tracking")
        self._below_threshold_since: float = -1.0
        self._alert_active = False

        # Smoothing buffer: average over last N scores to avoid jitter
        self._score_history: deque = deque(maxlen=10)

    # ────────────────────────────────────────────────────
    #  PUBLIC API
    # ────────────────────────────────────────────────────

    def update(
        self,
        gaze_direction: str,
        head_pose: str,
        ear: float,
        eyes_closed: bool,
        new_blink: bool,
        has_face: bool = True,
    ) -> dict:
        """
        Call once per frame with the outputs from EyeTracker.

        Returns a dict with:
            attention_score : int   (0-100)
            status          : str   ("Attentive" / "Distracted" / "Sleepy")
            alert           : bool  (True → show warning)
            blink_rate      : float (blinks per minute, rolling window)
        """
        now = time.time()

        # ── Track blink times ──────────────────────────────
        if new_blink:
            self._blink_times.append(now)
        # Remove blink timestamps older than our rolling window
        while self._blink_times and now - self._blink_times[0] > WINDOW_SECONDS:
            self._blink_times.popleft()

        # ── Track eyes-closed frames ───────────────────────
        if eyes_closed:
            self._closed_frame_count += 1
        else:
            self._closed_frame_count = 0

        if not has_face:
            self._score_history.clear()
            raw = 0.0
            smooth_score = 0
            status = "No Face"
            alert = True
            
            # Reset alert timer when no face so a real face gets 5s grace period again
            self._below_threshold_since = -1.0
            self._alert_active = True
        elif self._closed_frame_count >= SLEEP_EAR_FRAMES:
            self._score_history.clear()
            raw = 0.0
            smooth_score = 0
            status = "Sleepy"
            alert = True
            
            # Ensure the alert hits instantly instead of waiting 5s
            self._below_threshold_since = -1.0
            self._alert_active = True
        else:
            # ── Component scores (each 0-1) ─────────
            gaze_score  = self._score_gaze(gaze_direction)
            blink_score = self._score_blink()
            head_score  = self._score_head(head_pose)

            # ── Combine into raw score ─────────────────────────
            raw = (gaze_score * GAZE_WEIGHT
                 + blink_score * BLINK_WEIGHT
                 + head_score  * HEAD_WEIGHT) * 100

            # ── Smooth score ───────────────────────────────────
            self._score_history.append(raw)
            smooth_score = int(sum(self._score_history) / len(self._score_history))
            smooth_score = max(0, min(100, smooth_score))   # clamp to [0, 100]

            # ── Status label ──────────────────────────────────
            status = self._determine_status(smooth_score, gaze_direction)

            # ── Alert logic ───────────────────────────────────
            alert = self._check_alert(smooth_score, now)

        blink_rate: float = self._current_blink_rate(now)
        blink_rate_rounded: float = int(blink_rate * 10) / 10.0

        return {
            "attention_score": smooth_score,
            "status":          status,
            "alert":           alert,
            "blink_rate":      blink_rate_rounded,
            "gaze_direction":  gaze_direction,
            "head_pose":       head_pose,
        }

    # ────────────────────────────────────────────────────
    #  COMPONENT SCORE HELPERS
    # ────────────────────────────────────────────────────

    @staticmethod
    def _score_gaze(gaze: str) -> float:
        """
        Map gaze direction to a 0-1 attention contribution.
        Center = full attention; looking away = partial; Down = low.
        """
        mapping = {
            "Center": 1.00,
            "Left":   0.30,
            "Right":  0.30,
            "Down":   0.20,
        }
        return mapping.get(gaze, 0.5)

    def _score_blink(self) -> float:
        """
        Penalise extremely high blink rates (often means sleepiness)
        and reward normal rates.
        Returns a 0-1 score.
        """
        now   = time.time()
        rate  = self._current_blink_rate(now)

        # If eyes have been closed for > SLEEP_EAR_FRAMES → very low score
        if self._closed_frame_count >= SLEEP_EAR_FRAMES:
            return 0.0

        # Ideal blink rate is around 15 bpm
        # Too low (< 5) → staring / eye strain, slight penalty
        # Too high (> 25) → sleepy, big penalty
        if rate <= NORMAL_BLINK_RATE:
            # Scale 0→15 as 0.6→1.0
            return 0.6 + (rate / NORMAL_BLINK_RATE) * 0.4
        else:
            # Scale 15→30 as 1.0→0.2
            excess = min(rate - NORMAL_BLINK_RATE, MAX_BLINK_RATE - NORMAL_BLINK_RATE)
            return max(0.2, 1.0 - (excess / (MAX_BLINK_RATE - NORMAL_BLINK_RATE)) * 0.8)

    @staticmethod
    def _score_head(pose: str) -> float:
        """Map head pose to 0-1 contribution."""
        mapping = {
            "Forward": 1.00,
            "Left":    0.40,
            "Right":   0.40,
            "Down":    0.25,
        }
        return mapping.get(pose, 0.5)

    # ────────────────────────────────────────────────────
    #  STATUS LABEL
    # ────────────────────────────────────────────────────

    def _determine_status(self, score: int, gaze: str) -> str:
        """
        Three states:
          Sleepy    → eyes closed for extended period
          Distracted→ score below 55 or gaze not centered
          Attentive → everything else
        """
        if self._closed_frame_count >= SLEEP_EAR_FRAMES:
            return "Sleepy"
        if score < 55 or gaze in ("Left", "Right", "Down"):
            return "Distracted"
        return "Attentive"

    # ────────────────────────────────────────────────────
    #  ALERT SYSTEM
    # ────────────────────────────────────────────────────

    def _check_alert(self, score: int, now: float) -> bool:
        """
        Returns True if score has been below ALERT_THRESHOLD
        for at least ALERT_DURATION seconds.
        """
        if score < ALERT_THRESHOLD:
            if self._below_threshold_since < 0:
                self._below_threshold_since = now
            elif now - self._below_threshold_since >= ALERT_DURATION:
                self._alert_active = True
                return True
        else:
            # Score recovered → reset everything
            self._below_threshold_since = -1.0
            self._alert_active          = False
        return False

    # ────────────────────────────────────────────────────
    #  BLINK RATE
    # ────────────────────────────────────────────────────

    def _current_blink_rate(self, now: float) -> float:
        """Blinks per minute using the rolling window."""
        if not self._blink_times:
            return 0.0
        elapsed = now - self._blink_times[0]
        if elapsed < 1.0:
            return 0.0
        return len(self._blink_times) / elapsed * 60.0
