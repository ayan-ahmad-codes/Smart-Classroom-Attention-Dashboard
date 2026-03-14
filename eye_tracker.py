"""
eye_tracker.py
--------------
Face mesh, eye landmark extraction, EAR blink detection,
gaze direction, and head-pose estimation.

Updated to use the MediaPipe Tasks API (mediapipe >= 0.10),
which replaced the old `mp.solutions.face_mesh` interface.
"""

import os
import urllib.request

import cv2
import numpy as np

# ── New Tasks-API imports ─────────────────────────────────────────────────────
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)
from mediapipe.tasks.python.core.base_options import BaseOptions


# ─────────────────────────────────────────────────────────────────────────────
#  Model file  (downloaded automatically on first run, ~3 MB)
# ─────────────────────────────────────────────────────────────────────────────

MODEL_FILENAME = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)


def ensure_model_file() -> str:
    """
    Returns the local path to face_landmarker.task.
    Downloads it from Google if not already present.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, MODEL_FILENAME)

    if not os.path.isfile(model_path):
        print(f"[INFO] Model file not found. Downloading ({MODEL_FILENAME}) ...")
        print("       This only happens once (~3 MB).")
        try:
            urllib.request.urlretrieve(MODEL_URL, model_path)
            print(f"[INFO] Model saved → {model_path}")
        except Exception as e:
            raise RuntimeError(
                "\n[ERROR] Could not download the MediaPipe model file.\n"
                "        Please download it manually from:\n"
                f"        {MODEL_URL}\n"
                f"        and save it as '{MODEL_FILENAME}' in the same folder as eye_tracker.py.\n"
                f"        Original error: {e}"
            )
    return model_path


# ─────────────────────────────────────────────────────────────────────────────
#  Landmark indices  (same 468-point mesh as before)
# ─────────────────────────────────────────────────────────────────────────────

LEFT_EYE  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33,  7,  163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

# 6-point subsets used for the EAR formula
LEFT_EAR_POINTS  = [362, 385, 387, 263, 373, 380]
RIGHT_EAR_POINTS = [33,  160, 158, 133, 153, 144]

# Iris centre landmarks (available when refine_landmarks=True)
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Head-pose anchor points: nose tip, chin, mouth corners, face edges
HEAD_POSE_POINTS = [1, 9, 57, 130, 287, 359]


class EyeTracker:
    """
    Wraps MediaPipe FaceLandmarker (Tasks API) and exposes
    easy-to-use methods for eye and gaze tracking.
    """

    def __init__(self, ear_threshold: float = 0.22, consec_frames: int = 3):
        """
        Parameters
        ----------
        ear_threshold  : EAR below this → eye considered closed  (default 0.22)
        consec_frames  : frames of closed eye needed to register a blink (default 3)
        """
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames

        # ── Build the landmarker ──────────────────────────────────────────────
        model_path = ensure_model_file()

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,   # single-frame (synchronous) mode
            num_faces=1,
            min_face_detection_confidence=0.6,
            min_face_presence_confidence=0.6,
            min_tracking_confidence=0.6,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

        # ── State ─────────────────────────────────────────────────────────────
        self.blink_counter   = 0
        self.frame_counter   = 0   # consecutive closed-eye frames
        self.gaze_direction  = "Center"
        self.head_pose       = "Forward"
        self._last_pts       = None   # last pixel-coordinate array (N,2)

    # ─────────────────────────────────────────────────────────────────────────
    #  MAIN PROCESS METHOD
    # ─────────────────────────────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray):
        """
        Detect face and compute all eye metrics for one BGR frame.

        Returns
        -------
        has_face : bool   – True when a face is detected
        ear      : float  – combined Eye Aspect Ratio
        is_blink : bool   – True on the frame a blink is first registered
        """
        h, w = frame.shape[:2]

        # MediaPipe needs RGB
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = _to_mp_image(rgb)

        result   = self.landmarker.detect(mp_image)

        ear      = 0.0
        is_blink = False
        has_face = bool(result.face_landmarks)

        if has_face:
            # result.face_landmarks[0] → list of NormalizedLandmark
            raw = result.face_landmarks[0]
            pts = _lm_to_pixels(raw, h, w)
            self._last_pts = pts

            ear      = self._compute_ear(pts)
            is_blink = self._update_blink_state(ear)

            self.gaze_direction = self._compute_gaze(pts)
            self.head_pose      = self._compute_head_pose(pts, frame)
        else:
            self._last_pts = None

        return has_face, ear, is_blink

    # ─────────────────────────────────────────────────────────────────────────
    #  EAR  (Eye Aspect Ratio)
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_ear(self, pts: np.ndarray) -> float:
        left  = _ear_single(pts, LEFT_EAR_POINTS)
        right = _ear_single(pts, RIGHT_EAR_POINTS)
        return (left + right) / 2.0

    # ─────────────────────────────────────────────────────────────────────────
    #  BLINK STATE MACHINE
    # ─────────────────────────────────────────────────────────────────────────

    def _update_blink_state(self, ear: float) -> bool:
        is_blink = False
        if ear < self.ear_threshold:
            self.frame_counter += 1
        else:
            if self.frame_counter >= self.consec_frames:
                self.blink_counter += 1
                is_blink = True
            self.frame_counter = 0
        return is_blink

    # ─────────────────────────────────────────────────────────────────────────
    #  GAZE DIRECTION
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_gaze(self, pts: np.ndarray) -> str:
        """
        Estimate gaze by how far the iris centre is from each edge of the eye.
        Ratio 0 = iris far left; ratio 1 = iris far right.
        """
        try:
            # Left eye
            l_iris_cx = float(np.mean([pts[i][0] for i in LEFT_IRIS]))
            l_xmin    = float(min(pts[i][0] for i in LEFT_EYE))
            l_xmax    = float(max(pts[i][0] for i in LEFT_EYE))
            l_ymin    = float(min(pts[i][1] for i in LEFT_EYE))
            l_ymax    = float(max(pts[i][1] for i in LEFT_EYE))
            l_ratio_x = (l_iris_cx - l_xmin) / (l_xmax - l_xmin + 1e-6)

            # Right eye
            r_iris_cx = float(np.mean([pts[i][0] for i in RIGHT_IRIS]))
            r_xmin    = float(min(pts[i][0] for i in RIGHT_EYE))
            r_xmax    = float(max(pts[i][0] for i in RIGHT_EYE))
            r_ratio_x = (r_iris_cx - r_xmin) / (r_xmax - r_xmin + 1e-6)

            avg_x = (l_ratio_x + r_ratio_x) / 2.0

            # Vertical: look-down check
            l_iris_cy = float(np.mean([pts[i][1] for i in LEFT_IRIS]))
            l_ratio_y = (l_iris_cy - l_ymin) / (l_ymax - l_ymin + 1e-6)

            if l_ratio_y > 0.75:
                return "Down"
            elif avg_x < 0.38:
                return "Right"   # mirrored frame → swap
            elif avg_x > 0.62:
                return "Left"
            else:
                return "Center"
        except Exception:
            return "Center"

    # ─────────────────────────────────────────────────────────────────────────
    #  HEAD POSE
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_head_pose(self, pts: np.ndarray, frame: np.ndarray) -> str:
        """Rough head-orientation via solvePnP. Returns Forward/Left/Right/Down."""
        h, w = frame.shape[:2]

        # Generic 3-D face model (mm)
        model_3d = np.array([
            (0.0,    0.0,    0.0 ),   # nose tip
            (0.0,  -63.6,  -12.5),   # chin
            (-43.3, 32.7,  -26.0),   # left mouth corner
            ( 43.3, 32.7,  -26.0),   # right mouth corner
            (-28.9,-28.9,  -24.1),   # left eye outer
            ( 28.9,-28.9,  -24.1),   # right eye outer
        ], dtype=np.float64)

        image_2d = np.array([pts[i] for i in HEAD_POSE_POINTS], dtype=np.float64)

        focal      = float(w)
        cam_matrix = np.array([[focal, 0,     w / 2],
                                [0,     focal, h / 2],
                                [0,     0,     1    ]], dtype=np.float64)
        dist = np.zeros((4, 1), dtype=np.float64)

        ok, rot_vec, _ = cv2.solvePnP(
            model_3d, image_2d, cam_matrix, dist, flags=cv2.SOLVEPNP_SQPNP
        )
        if not ok:
            return "Forward"

        rot_mat, _ = cv2.Rodrigues(rot_vec)
        angles, *_ = cv2.RQDecomp3x3(rot_mat)
        yaw, pitch = angles[1], angles[0]

        if pitch < -10:
            return "Down"
        elif yaw < -15:
            return "Left"
        elif yaw > 15:
            return "Right"
        else:
            return "Forward"

    # ─────────────────────────────────────────────────────────────────────────
    #  DRAWING
    # ─────────────────────────────────────────────────────────────────────────

    def draw_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """Draw eye contours and iris dots onto the frame."""
        if self._last_pts is None:
            return frame
        pts = self._last_pts
        for idx in LEFT_EYE + RIGHT_EYE:
            cv2.circle(frame, (int(pts[idx][0]), int(pts[idx][1])), 1, (0, 255, 0), -1)
        for idx in LEFT_IRIS + RIGHT_IRIS:
            cv2.circle(frame, (int(pts[idx][0]), int(pts[idx][1])), 2, (0, 100, 255), -1)
        return frame

    # ─────────────────────────────────────────────────────────────────────────
    #  UTILITY
    # ─────────────────────────────────────────────────────────────────────────

    def get_blink_count(self) -> int:
        return self.blink_counter

    def reset_blink_count(self):
        self.blink_counter = 0

    def eyes_closed(self) -> bool:
        return self.frame_counter >= self.consec_frames

    def close(self):
        """Release MediaPipe resources."""
        self.landmarker.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _lm_to_pixels(landmarks, h: int, w: int) -> np.ndarray:
    """Convert NormalizedLandmark list → pixel (x, y) array of shape (N, 2)."""
    return np.array(
        [(lm.x * w, lm.y * h) for lm in landmarks],
        dtype=np.float32,
    )


def _to_mp_image(rgb_array: np.ndarray):
    """Wrap a numpy RGB array in a mediapipe.Image object."""
    import mediapipe as mp
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_array)


def _ear_single(pts: np.ndarray, indices: list) -> float:
    """EAR for one eye given 6 landmark indices."""
    p1, p2, p3, p4, p5, p6 = [pts[i] for i in indices]
    A = np.linalg.norm(p2 - p6)
    B = np.linalg.norm(p3 - p5)
    C = np.linalg.norm(p1 - p4)
    return (A + B) / (2.0 * C + 1e-6)