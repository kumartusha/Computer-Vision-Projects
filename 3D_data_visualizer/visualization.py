"""
visualization.py — 3D data renderer using pure OpenCV + NumPy.

Instead of opening a browser (Plotly) or requiring OpenGL drivers, we:
  1. Generate 3D geometry (clusters, DNA helix, torus ring, scattered nodes).
  2. Apply rotation / zoom / translation matrices each frame.
  3. Perspective-project every point to 2D.
  4. Depth-sort and render with neon glow — directly onto the webcam frame.

The result is a holographic overlay that responds in real-time to gestures.
"""

import numpy as np
import cv2
from utils import NEON_PALETTES, get_glow_color


class DataVisualizer3D:
    """Interactive 3D scatter-plot renderer with perspective projection."""

    def __init__(self, width: int = 640, height: int = 480):
        self.width  = width
        self.height = height
        self.screen_center = np.array([width // 2, height // 2], dtype=np.float64)
        self.focal_length  = 500.0

        # ── Transform state ───────────────────────────────────────────
        self.rotation    = np.array([0.35, 0.0, 0.0])   # pitch, yaw, roll
        self.zoom        = 1.0
        self.translation = np.array([0.0, 0.0, 0.0])

        # Targets (we lerp towards these for buttery-smooth motion)
        self.target_rotation    = self.rotation.copy()
        self.target_zoom        = 1.0
        self.target_translation = self.translation.copy()
        self.auto_rotate_speed  = 0.005   # radians/frame

        # ── Animation ─────────────────────────────────────────────────
        self.time_offset   = 0.0
        self.palette_index = 0

        # ── Generate data ─────────────────────────────────────────────
        self.points      = self._generate_data()
        self.edges        = self._compute_edges(threshold=1.5)
        self.point_sizes  = np.random.uniform(2, 6, len(self.points))

    # ══════════════════════════════════════════════════════════════════
    #  DATA GENERATION
    # ══════════════════════════════════════════════════════════════════

    def _generate_data(self) -> np.ndarray:
        """Build the 3D dataset: spherical clusters + DNA helix + torus."""
        pts: list[list[float]] = []

        # ── Cluster A — sphere of nodes ───────────────────────────────
        for _ in range(50):
            theta = np.random.uniform(0, 2 * np.pi)
            phi   = np.random.uniform(0, np.pi)
            r     = np.random.uniform(0.5, 1.5)
            pts.append([
                r * np.sin(phi) * np.cos(theta) - 2.5,
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi),
            ])

        # ── Cluster B — offset sphere ────────────────────────────────
        for _ in range(40):
            theta = np.random.uniform(0, 2 * np.pi)
            phi   = np.random.uniform(0, np.pi)
            r     = np.random.uniform(0.3, 1.2)
            pts.append([
                r * np.sin(phi) * np.cos(theta) + 2.5,
                r * np.sin(phi) * np.sin(theta) + 1.0,
                r * np.cos(phi),
            ])

        # ── DNA double helix ─────────────────────────────────────────
        for i in range(60):
            t  = i * 0.15
            y  = t * 0.3 - 3.0
            pts.append([ np.cos(t) * 0.8,          y,  np.sin(t) * 0.8])
            pts.append([ np.cos(t + np.pi) * 0.8,  y,  np.sin(t + np.pi) * 0.8])

        # ── Torus ring ───────────────────────────────────────────────
        R_major = 2.0
        r_minor = 0.4
        for i in range(45):
            theta = i * 2 * np.pi / 45
            phi   = np.random.uniform(0, 2 * np.pi)
            pts.append([
                (R_major + r_minor * np.cos(phi)) * np.cos(theta),
                r_minor * np.sin(phi) + 3.0,
                (R_major + r_minor * np.cos(phi)) * np.sin(theta),
            ])

        # ── Random floating particles ────────────────────────────────
        for _ in range(35):
            pts.append([
                np.random.uniform(-4, 4),
                np.random.uniform(-4, 4),
                np.random.uniform(-3, 3),
            ])

        return np.array(pts, dtype=np.float64)

    def _compute_edges(self, threshold: float = 1.5):
        """Pre-compute edges between points closer than *threshold*."""
        edges = []
        n = len(self.points)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(self.points[i] - self.points[j])
                if d < threshold:
                    edges.append((i, j, d))
        return edges

    # ══════════════════════════════════════════════════════════════════
    #  3D → 2D PROJECTION
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _rotation_matrix(angles):
        """Euler rotation matrix (pitch, yaw, roll)."""
        p, y, r = angles
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        cr, sr = np.cos(r), np.sin(r)

        Rx = np.array([[1, 0, 0],   [0, cp, -sp], [0, sp, cp]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0],    [-sy, 0, cy]])
        Rz = np.array([[cr, -sr, 0],[sr, cr, 0],  [0, 0, 1]])
        return Rz @ Ry @ Rx

    def _project(self):
        """Return (x_2d, y_2d, depth) arrays for all points."""
        R = self._rotation_matrix(self.rotation)
        rotated = self.points @ R.T
        rotated *= self.zoom
        rotated += self.translation

        # Camera sits at z=0 looking down +z — push scene forward
        camera_dist = 10.0
        rotated[:, 2] += camera_dist

        z = np.clip(rotated[:, 2], 0.1, None)
        x2 = self.focal_length * rotated[:, 0] / z + self.screen_center[0]
        y2 = self.focal_length * rotated[:, 1] / z + self.screen_center[1]
        return x2, y2, z

    # ══════════════════════════════════════════════════════════════════
    #  GESTURE → TRANSFORM UPDATE
    # ══════════════════════════════════════════════════════════════════

    def update(self, gesture_data: dict):
        """Apply gesture controls and advance animation."""
        gesture  = gesture_data["gesture"]
        movement = gesture_data["movement"]

        # Gentle constant auto-rotation
        self.target_rotation[1] += self.auto_rotate_speed

        # ── Gesture mapping ───────────────────────────────────────────
        if gesture == "PINCH":
            # Pinch + move hand up/down → zoom in/out
            self.target_zoom -= movement[1] * 0.005
            self.target_zoom = float(np.clip(self.target_zoom, 0.3, 3.5))

        elif gesture == "OPEN PALM":
            # Reset everything
            self.target_rotation    = np.array([0.35, self.target_rotation[1], 0.0])
            self.target_zoom        = 1.0
            self.target_translation = np.array([0.0, 0.0, 0.0])

        elif gesture == "FIST":
            # Drag / translate the graph
            self.target_translation[0] += movement[0] * 0.018
            self.target_translation[1] += movement[1] * 0.018

        elif gesture == "POINT":
            # Rotate graph based on finger movement
            self.target_rotation[1] += movement[0] * 0.007   # yaw
            self.target_rotation[0] += movement[1] * 0.007   # pitch

        elif gesture == "SWIPE LEFT":
            self.palette_index = (self.palette_index + 1) % len(NEON_PALETTES)

        elif gesture == "SWIPE RIGHT":
            self.palette_index = (self.palette_index - 1) % len(NEON_PALETTES)

        # ── Smooth interpolation (lerp) ───────────────────────────────
        lerp = 0.12
        self.rotation    += (self.target_rotation    - self.rotation)    * lerp
        self.zoom        += (self.target_zoom        - self.zoom)        * lerp
        self.translation += (self.target_translation - self.translation) * lerp

        self.time_offset += 0.025

    # ══════════════════════════════════════════════════════════════════
    #  RENDERING
    # ══════════════════════════════════════════════════════════════════

    def render(self, frame):
        """Draw the 3D scene onto *frame* (modifies in-place)."""
        x2, y2, depth = self._project()
        palette = NEON_PALETTES[self.palette_index]
        n_colors = len(palette)

        # Render back-to-front
        order = np.argsort(-depth)

        self._draw_grid(frame)
        self._draw_edges(frame, x2, y2, depth, palette, n_colors)
        self._draw_nodes(frame, x2, y2, depth, palette, n_colors, order)

    # ── Sub-renderers ─────────────────────────────────────────────────

    def _draw_grid(self, frame):
        """Perspective grid on the 'floor' plane."""
        R = self._rotation_matrix(self.rotation)
        grid_y = 4.5
        cam_d  = 10.0

        for i in range(-5, 6):
            for start_3d, end_3d in [
                ([i, grid_y, -5], [i, grid_y, 5]),
                ([-5, grid_y, i], [5, grid_y, i]),
            ]:
                s = R @ (np.array(start_3d) * self.zoom + self.translation)
                e = R @ (np.array(end_3d)   * self.zoom + self.translation)
                s[2] += cam_d
                e[2] += cam_d
                if s[2] < 0.1 or e[2] < 0.1:
                    continue
                sx = int(self.focal_length * s[0] / s[2] + self.screen_center[0])
                sy = int(self.focal_length * s[1] / s[2] + self.screen_center[1])
                ex = int(self.focal_length * e[0] / e[2] + self.screen_center[0])
                ey = int(self.focal_length * e[1] / e[2] + self.screen_center[1])
                cv2.line(frame, (sx, sy), (ex, ey), (40, 55, 50), 1, cv2.LINE_AA)

    def _draw_edges(self, frame, x2, y2, depth, palette, n_colors):
        for i, j, _ in self.edges:
            xi, yi = int(x2[i]), int(y2[i])
            xj, yj = int(x2[j]), int(y2[j])
            if not self._on_screen(xi, yi) and not self._on_screen(xj, yj):
                continue
            avg_d = (depth[i] + depth[j]) / 2.0
            alpha = float(np.clip(1.0 - (avg_d - 8) / 12, 0.08, 0.45))
            ci = int((i + self.time_offset * 4) % n_colors)
            color = tuple(max(0, min(255, int(c * alpha))) for c in palette[ci])
            cv2.line(frame, (xi, yi), (xj, yj), color, 1, cv2.LINE_AA)

    def _draw_nodes(self, frame, x2, y2, depth, palette, n_colors, order):
        for idx in order:
            x = int(x2[idx])
            y = int(y2[idx])
            d = depth[idx]
            if not self._on_screen(x, y):
                continue

            # Size shrinks with depth
            sz = max(1, int(self.point_sizes[idx] * 8.0 / d))

            # Animated colour interpolation
            phase    = (idx * 0.1 + self.time_offset) % n_colors
            ci       = int(phase)
            ci_next  = (ci + 1) % n_colors
            t        = phase - ci
            c1       = np.array(palette[ci],      dtype=np.float64)
            c2       = np.array(palette[ci_next],  dtype=np.float64)
            base     = c1 * (1 - t) + c2 * t

            # Depth-based brightness
            bright = float(np.clip(1.0 - (d - 8) / 15.0, 0.2, 1.0))
            color  = tuple(max(0, min(255, int(c * bright))) for c in base)

            # Layered glow
            if sz >= 2:
                g1 = tuple(max(0, min(255, int(c * 0.12))) for c in color)
                g2 = tuple(max(0, min(255, int(c * 0.35))) for c in color)
                cv2.circle(frame, (x, y), sz + 5, g1, -1, cv2.LINE_AA)
                cv2.circle(frame, (x, y), sz + 2, g2, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), sz, color, -1, cv2.LINE_AA)
            # Bright core highlight
            if sz >= 3:
                cv2.circle(frame, (x, y), max(1, sz // 3),
                           (255, 255, 255), -1, cv2.LINE_AA)

    # ── Utility ───────────────────────────────────────────────────────

    def _on_screen(self, x: int, y: int, margin: int = 60) -> bool:
        return -margin < x < self.width + margin and \
               -margin < y < self.height + margin

    def reset(self):
        """Animate back to the default view."""
        self.target_rotation    = np.array([0.35, 0.0, 0.0])
        self.target_zoom        = 1.0
        self.target_translation = np.array([0.0, 0.0, 0.0])
