"""
gestures.py — Gesture recognition engine.

Analyses HandTracker output each frame and classifies the current gesture:

  POINT        — index finger extended, others curled  → rotate graph
  PINCH        — thumb + index close together          → zoom graph
  FIST         — all fingers curled                    → drag graph
  OPEN PALM    — all fingers extended                  → reset graph
  SWIPE LEFT   — fast leftward hand motion             → previous palette
  SWIPE RIGHT  — fast rightward hand motion            → next palette
  NONE         — no clear gesture detected
"""

import time
import numpy as np
from collections import Counter
from utils import MotionSmoother

# ── Gesture labels ────────────────────────────────────────────────────
GESTURE_NONE        = "NONE"
GESTURE_PINCH       = "PINCH"
GESTURE_OPEN_PALM   = "OPEN PALM"
GESTURE_FIST        = "FIST"
GESTURE_POINT       = "POINT"
GESTURE_SWIPE_LEFT  = "SWIPE LEFT"
GESTURE_SWIPE_RIGHT = "SWIPE RIGHT"


class GestureRecognizer:
    """Stateful gesture classifier with smoothing and debouncing."""

    def __init__(self):
        # ── Thresholds ────────────────────────────────────────────────
        self.pinch_threshold  = 45   # px — thumb↔index distance for "pinch"
        self.swipe_threshold  = 70   # px/frame — min horizontal speed for swipe
        self.swipe_cooldown   = 0.6  # seconds between consecutive swipe triggers

        # ── Internal state ────────────────────────────────────────────
        self.prev_center     = None
        self.last_swipe_time = 0.0
        self._history: list[str]  = []
        self._history_size        = 5

        # Smoothers for derived signals
        self._pinch_sm    = MotionSmoother(alpha=0.4)
        self._movement_sm = MotionSmoother(alpha=0.3)

    # ── Main API ──────────────────────────────────────────────────────

    def recognize(self, tracker) -> dict:
        """Classify the current gesture.

        Returns a dict with keys:
            gesture        – one of the GESTURE_* constants
            pinch_distance – smoothed thumb↔index distance (or None)
            hand_center    – smoothed hand centre in pixels (or None)
            movement       – (dx, dy) movement delta
            confidence     – rough 0-1 confidence score
        """
        result = {
            "gesture":        GESTURE_NONE,
            "pinch_distance": None,
            "hand_center":    None,
            "movement":       np.array([0.0, 0.0]),
            "confidence":     0.0,
        }

        if not tracker.hand_detected:
            self.prev_center = None
            return result

        # ── Gather data ───────────────────────────────────────────────
        finger_states = tracker.get_finger_states()
        pinch_dist    = tracker.get_pinch_distance()
        hand_center   = tracker.get_hand_center()

        if hand_center is None:
            return result
        result["hand_center"] = hand_center

        # ── Compute movement delta ────────────────────────────────────
        movement = np.array([0.0, 0.0])
        if self.prev_center is not None:
            raw = hand_center - self.prev_center
            movement = self._movement_sm.update(raw)
        self.prev_center = hand_center.copy()
        result["movement"] = movement

        # ── 1. Swipe (highest priority, time-gated) ───────────────────
        now = time.time()
        if now - self.last_swipe_time > self.swipe_cooldown:
            if abs(movement[0]) > self.swipe_threshold:
                result["gesture"] = (GESTURE_SWIPE_LEFT
                                     if movement[0] < 0
                                     else GESTURE_SWIPE_RIGHT)
                result["confidence"] = min(
                    abs(movement[0]) / (self.swipe_threshold * 2), 1.0)
                self.last_swipe_time = now
                self._push(result["gesture"])
                return result

        # ── 2. Pinch ──────────────────────────────────────────────────
        if pinch_dist is not None:
            smooth_pinch = self._pinch_sm.update([pinch_dist])[0]
            result["pinch_distance"] = smooth_pinch

            if smooth_pinch < self.pinch_threshold:
                result["gesture"]    = GESTURE_PINCH
                result["confidence"] = 1.0 - (smooth_pinch / self.pinch_threshold)
                self._push(result["gesture"])
                return result

        # ── 3. Open palm (≥ 4 fingers extended) ───────────────────────
        extended = sum(finger_states)
        if extended >= 4:
            result["gesture"]    = GESTURE_OPEN_PALM
            result["confidence"] = extended / 5.0
            self._push(result["gesture"])
            return result

        # ── 4. Closed fist (0-1 fingers, index not up) ────────────────
        if extended <= 1 and not finger_states[1]:
            result["gesture"]    = GESTURE_FIST
            result["confidence"] = 1.0 - extended / 5.0
            self._push(result["gesture"])
            return result

        # ── 5. Point (index up, ≤ 2 total) ───────────────────────────
        if finger_states[1] and extended <= 2:
            result["gesture"]    = GESTURE_POINT
            result["confidence"] = 0.85
            self._push(result["gesture"])
            return result

        # ── Fallback ──────────────────────────────────────────────────
        self._push(GESTURE_NONE)
        return result

    # ── Helpers ───────────────────────────────────────────────────────

    def _push(self, gesture: str):
        self._history.append(gesture)
        if len(self._history) > self._history_size:
            self._history.pop(0)

    def get_stable_gesture(self) -> str:
        """Most frequent gesture in the recent window (for display)."""
        if not self._history:
            return GESTURE_NONE
        return Counter(self._history).most_common(1)[0][0]
