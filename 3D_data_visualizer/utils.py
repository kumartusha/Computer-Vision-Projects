"""
utils.py — Utility classes for motion smoothing, FPS tracking, and visual effects.

Provides:
  - MotionSmoother   : Exponential moving average to kill hand-tracking jitter.
  - FPSCounter       : Rolling-window frames-per-second measurement.
  - HandTrail        : Stores recent hand positions and draws a fading trail.
  - Neon palettes    : Curated BGR colour sets for the 3D graph.
  - HUD helpers      : Glow circles, scanlines, text with shadow.
"""

import time
import numpy as np
import cv2


# ══════════════════════════════════════════════════════════════════════
#  MOTION SMOOTHING
# ══════════════════════════════════════════════════════════════════════

class MotionSmoother:
    """Exponential moving average smoother to reduce hand-tracking jitter.

    Lower alpha → smoother output but more input lag.
    Higher alpha → more responsive but jittery.
    """

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        """Feed a new raw value; returns the smoothed result."""
        if self.value is None:
            self.value = np.array(new_value, dtype=np.float64)
        else:
            self.value = (self.alpha * np.array(new_value, dtype=np.float64)
                          + (1 - self.alpha) * self.value)
        return self.value.copy()

    def reset(self):
        self.value = None


# ══════════════════════════════════════════════════════════════════════
#  FPS COUNTER
# ══════════════════════════════════════════════════════════════════════

class FPSCounter:
    """Rolling-window FPS tracker."""

    def __init__(self, window: int = 30):
        self._timestamps: list[float] = []
        self._window = window

    def tick(self):
        """Call once per frame."""
        self._timestamps.append(time.time())
        if len(self._timestamps) > self._window:
            self._timestamps.pop(0)

    def get_fps(self) -> float:
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════
#  HAND TRAIL
# ══════════════════════════════════════════════════════════════════════

class HandTrail:
    """Stores recent hand positions and renders a fading neon trail."""

    def __init__(self, max_length: int = 30):
        self.positions: list[tuple] = []
        self.max_length = max_length

    def add(self, pos):
        if pos is not None:
            self.positions.append((int(pos[0]), int(pos[1])))
            if len(self.positions) > self.max_length:
                self.positions.pop(0)

    def draw(self, frame, color=(0, 255, 255)):
        for i in range(1, len(self.positions)):
            alpha = i / len(self.positions)
            thickness = max(1, int(alpha * 3))
            c = tuple(int(v * alpha) for v in color)
            cv2.line(frame,
                     self.positions[i - 1],
                     self.positions[i],
                     c, thickness, cv2.LINE_AA)

    def clear(self):
        self.positions.clear()


# ══════════════════════════════════════════════════════════════════════
#  NEON COLOUR PALETTES  (BGR order for OpenCV)
# ══════════════════════════════════════════════════════════════════════

NEON_PALETTES = [
    # 0 — Cyan / Electric Blue
    [(255, 255, 0), (255, 200, 0), (255, 150, 0), (255, 255, 50), (200, 255, 0)],
    # 1 — Magenta / Violet
    [(255, 0, 255), (200, 50, 255), (255, 0, 200), (255, 100, 255), (220, 0, 180)],
    # 2 — Matrix Green
    [(100, 255, 0), (0, 255, 0), (50, 255, 50), (100, 200, 0), (0, 255, 100)],
    # 3 — Solar Orange
    [(0, 165, 255), (0, 100, 255), (0, 200, 255), (0, 140, 255), (50, 180, 255)],
    # 4 — Mixed Neon
    [(255, 0, 255), (255, 255, 0), (100, 255, 0), (0, 100, 255), (255, 0, 100)],
]

PALETTE_NAMES = ["CYAN", "MAGENTA", "MATRIX", "SOLAR", "MIXED"]


# ══════════════════════════════════════════════════════════════════════
#  DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════

def get_glow_color(base_color, intensity: float = 1.0):
    """Return a dimmed copy of *base_color* for layered glow effects."""
    return tuple(max(0, min(255, int(c * intensity))) for c in base_color)


def draw_glow_circle(frame, center, radius, color, layers: int = 4):
    """Draw a circle with concentric glow rings."""
    for i in range(layers, 0, -1):
        r = radius + i * 3
        glow = get_glow_color(color, 0.15 / i * 3)
        cv2.circle(frame, center, r, glow, -1, cv2.LINE_AA)
    cv2.circle(frame, center, radius, color, -1, cv2.LINE_AA)


def draw_hud_text(frame, text, pos, color=(0, 255, 255),
                  scale=0.6, thickness=1):
    """Draw text with a dark shadow underneath for readability."""
    # Shadow
    cv2.putText(frame, text, (pos[0] + 1, pos[1] + 1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thickness + 2, cv2.LINE_AA)
    # Foreground
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thickness, cv2.LINE_AA)


def draw_scanline_overlay(frame, intensity: float = 0.03):
    """Add subtle horizontal scanlines for a retro-futuristic look."""
    h = frame.shape[0]
    dim = int(255 * intensity)
    for y in range(0, h, 3):
        frame[y, :] = np.clip(
            frame[y, :].astype(np.int16) - dim, 0, 255
        ).astype(np.uint8)
