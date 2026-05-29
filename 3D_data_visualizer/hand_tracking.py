"""
hand_tracking.py — Real-time hand tracking using the MediaPipe Tasks API.

MediaPipe ≥ 0.10.14 removed the legacy ``mp.solutions`` interface.
This module uses the current ``mediapipe.tasks.python.vision.HandLandmarker``
and downloads the model file automatically on first run.
"""

import os
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from utils import MotionSmoother, draw_glow_circle


# ── Model download ────────────────────────────────────────────────────
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "hand_landmarker.task")


def _ensure_model():
    """Download the hand-landmarker model if it isn't cached locally."""
    if os.path.exists(_MODEL_PATH):
        return
    print("  ⬇ Downloading hand_landmarker model (~12 MB) …")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    print("  ✓ Model saved to", _MODEL_PATH)


# ── Hand skeleton topology ────────────────────────────────────────────
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (13, 17), (17, 18), (18, 19), (19, 20),
    # Palm base
    (0, 17),
]


class HandTracker:
    """Detects a single hand via the MediaPipe Tasks Hand Landmarker
    and exposes smoothed key-point positions."""

    # ── Landmark indices ──────────────────────────────────────────────
    WRIST       = 0
    THUMB_CMC   = 1
    THUMB_MCP   = 2
    THUMB_IP    = 3
    THUMB_TIP   = 4
    INDEX_MCP   = 5
    INDEX_PIP   = 6
    INDEX_DIP   = 7
    INDEX_TIP   = 8
    MIDDLE_MCP  = 9
    MIDDLE_PIP  = 10
    MIDDLE_DIP  = 11
    MIDDLE_TIP  = 12
    RING_MCP    = 13
    RING_PIP    = 14
    RING_DIP    = 15
    RING_TIP    = 16
    PINKY_MCP   = 17
    PINKY_PIP   = 18
    PINKY_DIP   = 19
    PINKY_TIP   = 20

    def __init__(self, max_hands=1, detection_conf=0.7, tracking_conf=0.7):
        _ensure_model()

        base_opts = mp_python.BaseOptions(
            model_asset_path=_MODEL_PATH,
        )
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

        # Smoothers for the three key points
        self._index_sm  = MotionSmoother(alpha=0.35)
        self._thumb_sm  = MotionSmoother(alpha=0.35)
        self._center_sm = MotionSmoother(alpha=0.25)

        # Per-frame state
        self._landmarks: list | None = None     # list[NormalizedLandmark]
        self.hand_detected: bool     = False
        self._frame_h: int           = 0
        self._frame_w: int           = 0
        self._frame_idx: int         = 0        # monotonic frame counter

    # ── Per-frame processing ──────────────────────────────────────────

    def process(self, frame) -> bool:
        """Run hand detection on *frame* (BGR). Returns True if a hand is found."""
        self._frame_h, self._frame_w = frame.shape[:2]

        # Convert BGR → RGB and wrap in mp.Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Timestamp must be strictly increasing (milliseconds)
        self._frame_idx += 1
        timestamp_ms = int(self._frame_idx * (1000 / 30))   # ~30 fps clock

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.hand_landmarks:
            self._landmarks = result.hand_landmarks[0]  # first hand
            self.hand_detected = True
        else:
            self._landmarks = None
            self.hand_detected = False

        return self.hand_detected

    # ── Landmark accessors ────────────────────────────────────────────

    def get_landmark_px(self, idx):
        """Return landmark *idx* as (x, y) in pixel coords, or None."""
        if self._landmarks is None:
            return None
        lm = self._landmarks[idx]
        return np.array([lm.x * self._frame_w, lm.y * self._frame_h])

    def get_landmark_norm(self, idx):
        """Return landmark *idx* as normalised (x, y, z), or None."""
        if self._landmarks is None:
            return None
        lm = self._landmarks[idx]
        return np.array([lm.x, lm.y, lm.z])

    # ── Smoothed key points ───────────────────────────────────────────

    def get_index_tip(self):
        raw = self.get_landmark_px(self.INDEX_TIP)
        return None if raw is None else self._index_sm.update(raw)

    def get_thumb_tip(self):
        raw = self.get_landmark_px(self.THUMB_TIP)
        return None if raw is None else self._thumb_sm.update(raw)

    def get_hand_center(self):
        """Midpoint of wrist ↔ middle-finger MCP, smoothed."""
        wrist   = self.get_landmark_px(self.WRIST)
        mid_mcp = self.get_landmark_px(self.MIDDLE_MCP)
        if wrist is None or mid_mcp is None:
            return None
        return self._center_sm.update((wrist + mid_mcp) / 2.0)

    # ── Finger-state detection ────────────────────────────────────────

    def get_finger_states(self) -> list[bool]:
        """[thumb, index, middle, ring, pinky] — True means extended."""
        if self._landmarks is None:
            return [False] * 5

        states: list[bool] = []

        # Thumb: tip further from MCP than IP is
        tip = self.get_landmark_norm(self.THUMB_TIP)
        ip  = self.get_landmark_norm(self.THUMB_IP)
        mcp = self.get_landmark_norm(self.THUMB_MCP)
        if tip is not None and ip is not None and mcp is not None:
            d_tip = np.linalg.norm(tip[:2] - mcp[:2])
            d_ip  = np.linalg.norm(ip[:2]  - mcp[:2])
            states.append(d_tip > d_ip * 1.1)
        else:
            states.append(False)

        # Other fingers: tip.y < pip.y  (lower y = higher on screen)
        tips = [self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]
        pips = [self.INDEX_PIP, self.MIDDLE_PIP, self.RING_PIP, self.PINKY_PIP]
        for t_id, p_id in zip(tips, pips):
            t = self.get_landmark_norm(t_id)
            p = self.get_landmark_norm(p_id)
            states.append(t[1] < p[1] if (t is not None and p is not None) else False)

        return states

    # ── Pinch distance ────────────────────────────────────────────────

    def get_pinch_distance(self):
        """Pixel distance between thumb tip and index tip."""
        thumb = self.get_landmark_px(self.THUMB_TIP)
        index = self.get_landmark_px(self.INDEX_TIP)
        if thumb is None or index is None:
            return None
        return float(np.linalg.norm(thumb - index))

    # ── Futuristic drawing ────────────────────────────────────────────

    def draw_landmarks(self, frame, color=(255, 255, 0)):
        """Render the hand skeleton with neon glow."""
        if self._landmarks is None:
            return
        h, w = frame.shape[:2]

        # Connections
        for (a, b) in HAND_CONNECTIONS:
            la = self._landmarks[a]
            lb = self._landmarks[b]
            sp = (int(la.x * w), int(la.y * h))
            ep = (int(lb.x * w), int(lb.y * h))
            # Outer glow
            cv2.line(frame, sp, ep,
                     tuple(max(0, int(c * 0.3)) for c in color),
                     4, cv2.LINE_AA)
            # Core
            cv2.line(frame, sp, ep, color, 1, cv2.LINE_AA)

        # Landmark dots
        fingertip_ids = {self.THUMB_TIP, self.INDEX_TIP,
                         self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP}
        for i, lm in enumerate(self._landmarks):
            px = (int(lm.x * w), int(lm.y * h))
            if i in fingertip_ids:
                draw_glow_circle(frame, px, 4, color, layers=3)
            else:
                cv2.circle(frame, px, 2, color, -1, cv2.LINE_AA)

    # ── Cleanup ───────────────────────────────────────────────────────

    def release(self):
        self._landmarker.close()
