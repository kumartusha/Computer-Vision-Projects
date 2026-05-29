"""
╔═══════════════════════════════════════════════════════════════════════╗
║          GESTURE CONTROLLED 3D DATA VISUALIZATION                    ║
║                                                                      ║
║   A Minority-Report-style holographic interface.                     ║
║   Your webcam feed becomes the backdrop; a glowing 3D data graph     ║
║   floats in front of you and responds to hand gestures in real time. ║
║                                                                      ║
║   Controls:                                                          ║
║     POINT (index finger)  → rotate the graph                         ║
║     PINCH (thumb + index) → zoom in / out                            ║
║     FIST                  → drag / translate                         ║
║     OPEN PALM             → reset view                               ║
║     SWIPE LEFT / RIGHT    → change colour palette                    ║
║     R key                 → reset view                               ║
║     Q / Esc               → quit                                     ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np

from hand_tracking import HandTracker
from gestures import GestureRecognizer
from visualization import DataVisualizer3D
from utils import (
    FPSCounter,
    HandTrail,
    draw_hud_text,
    draw_scanline_overlay,
    NEON_PALETTES,
    PALETTE_NAMES,
)


# ══════════════════════════════════════════════════════════════════════
#  HUD OVERLAY
# ══════════════════════════════════════════════════════════════════════

def draw_hud(frame, gesture_data, fps, viz):
    """Render the heads-up display: title, FPS, gesture, palette, help."""
    h, w = frame.shape[:2]
    gesture = gesture_data["gesture"]

    # ── Title bar ─────────────────────────────────────────────────────
    draw_hud_text(frame, "GESTURE CONTROL 3D",
                  (15, 32), color=(255, 255, 0), scale=0.75, thickness=2)
    draw_hud_text(frame, f"FPS {fps:.0f}",
                  (15, 58), color=(200, 220, 0), scale=0.48)

    # ── Gesture label (colour-coded) ──────────────────────────────────
    gesture_colors = {
        "NONE":        (100, 130, 120),
        "POINT":       (255, 200, 0),
        "PINCH":       (100, 255, 0),
        "FIST":        (0, 140, 255),
        "OPEN PALM":   (0, 255, 255),
        "SWIPE LEFT":  (255, 0, 255),
        "SWIPE RIGHT": (255, 0, 255),
    }
    g_color = gesture_colors.get(gesture, (200, 200, 200))
    draw_hud_text(frame, f"GESTURE: {gesture}",
                  (15, h - 22), color=g_color, scale=0.6, thickness=1)

    # ── Palette & zoom info (top-right) ───────────────────────────────
    pi = viz.palette_index
    draw_hud_text(frame, f"PALETTE: {PALETTE_NAMES[pi]}",
                  (w - 220, 32), color=NEON_PALETTES[pi][0], scale=0.5)
    draw_hud_text(frame, f"ZOOM: {viz.zoom:.1f}x",
                  (w - 220, 56), color=(200, 220, 0), scale=0.5)

    # ── Corner brackets ───────────────────────────────────────────────
    L = 35
    bc = (0, 150, 140)
    cv2.line(frame, (5, 5), (5 + L, 5),     bc, 1, cv2.LINE_AA)
    cv2.line(frame, (5, 5), (5, 5 + L),     bc, 1, cv2.LINE_AA)
    cv2.line(frame, (w-6, 5), (w-6-L, 5),   bc, 1, cv2.LINE_AA)
    cv2.line(frame, (w-6, 5), (w-6, 5+L),   bc, 1, cv2.LINE_AA)
    cv2.line(frame, (5, h-6), (5+L, h-6),   bc, 1, cv2.LINE_AA)
    cv2.line(frame, (5, h-6), (5, h-6-L),   bc, 1, cv2.LINE_AA)
    cv2.line(frame, (w-6, h-6), (w-6-L, h-6), bc, 1, cv2.LINE_AA)
    cv2.line(frame, (w-6, h-6), (w-6, h-6-L), bc, 1, cv2.LINE_AA)

    # ── Quick-help (bottom-right) ─────────────────────────────────────
    helps = [
        "POINT  -> Rotate",
        "PINCH  -> Zoom",
        "FIST   -> Drag",
        "PALM   -> Reset",
        "SWIPE  -> Colors",
    ]
    for i, txt in enumerate(helps):
        draw_hud_text(frame, txt,
                      (w - 195, h - 130 + i * 22),
                      color=(0, 120, 120), scale=0.4)


# ══════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════

def main():
    # ── Banner ────────────────────────────────────────────────────────
    print()
    print("=" * 56)
    print("   GESTURE CONTROLLED 3D DATA VISUALIZATION")
    print("=" * 56)
    print()
    print("  Starting webcam ...")
    print("  Controls:")
    print("    POINT  finger  →  rotate graph")
    print("    PINCH          →  zoom in / out")
    print("    FIST           →  drag graph")
    print("    OPEN PALM      →  reset view")
    print("    SWIPE L/R      →  change palette")
    print("    R key          →  reset   |   Q / Esc → quit")
    print()

    # ── Open webcam ───────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, first = cap.read()
    if not ret:
        print("[ERROR] Could not read from webcam!")
        cap.release()
        return
    h, w = first.shape[:2]
    print(f"  Camera resolution: {w}x{h}")
    print("  System ready — show your hand!\n")

    # ── Components ────────────────────────────────────────────────────
    tracker    = HandTracker(max_hands=1, detection_conf=0.7, tracking_conf=0.7)
    recognizer = GestureRecognizer()
    viz        = DataVisualizer3D(width=w, height=h)
    fps        = FPSCounter()
    trail      = HandTrail(max_length=25)

    # Default gesture state
    no_gesture = {
        "gesture": "NONE",
        "pinch_distance": None,
        "hand_center": None,
        "movement": np.array([0.0, 0.0]),
        "confidence": 0.0,
    }

    # ── Loop ──────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror for natural interaction
        frame = cv2.flip(frame, 1)

        # ── Cyberpunk tint ────────────────────────────────────────────
        frame = (frame.astype(np.float32) * 0.28).astype(np.uint8)
        tint = np.zeros_like(frame)
        tint[:, :, 0] = 18   # blue channel bump
        tint[:, :, 1] = 6    # green channel bump
        frame = cv2.add(frame, tint)

        # ── Hand tracking ─────────────────────────────────────────────
        hand_found = tracker.process(frame)

        if hand_found:
            gesture_data = recognizer.recognize(tracker)
            trail.add(gesture_data.get("hand_center"))
            tracker.draw_landmarks(frame, color=(255, 255, 0))
            trail.draw(frame, color=(255, 200, 0))

            # Pinch visual indicator
            if gesture_data["gesture"] == "PINCH":
                thumb = tracker.get_thumb_tip()
                index = tracker.get_index_tip()
                if thumb is not None and index is not None:
                    mid = ((thumb + index) / 2).astype(int)
                    cv2.circle(frame, tuple(mid), 10, (100, 255, 0), 2, cv2.LINE_AA)
        else:
            gesture_data = no_gesture

        # ── Update & render 3D scene ──────────────────────────────────
        viz.update(gesture_data)
        viz.render(frame)

        # ── Post-processing ───────────────────────────────────────────
        draw_scanline_overlay(frame, intensity=0.018)
        fps.tick()
        draw_hud(frame, gesture_data, fps.get_fps(), viz)

        # ── Display ───────────────────────────────────────────────────
        cv2.imshow("Gesture Controlled 3D Visualization", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):       # Q or Esc
            break
        elif key == ord("r"):           # R → reset view
            viz.reset()

    # ── Cleanup ───────────────────────────────────────────────────────
    tracker.release()
    cap.release()
    cv2.destroyAllWindows()
    print("\n  Shutdown complete.\n")


if __name__ == "__main__":
    main()
