"""
server.py — WebSocket bridge between Python hand tracking and Three.js.

Architecture
────────────
  Main thread    →  OpenCV webcam + MediaPipe hand tracking loop
  Daemon thread  →  WebSocket server  (ws://localhost:8765)
  Daemon thread  →  HTTP file server   (http://localhost:8080)

The browser page (viewer.html) connects via WebSocket and receives
gesture JSON at ~30 fps. A small OpenCV window shows the hand-tracking
overlay so you can see what the camera sees.

Usage
─────
  python3 server.py          # opens browser automatically
  Press Q or Esc in the tracking window to quit.
"""

import asyncio
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

import cv2
import numpy as np

from hand_tracking import HandTracker
from gestures import GestureRecognizer
from utils import (
    FPSCounter,
    HandTrail,
    draw_hud_text,
    draw_scanline_overlay,
    NEON_PALETTES,
    PALETTE_NAMES,
)

# ── Lazy-install websockets if missing ────────────────────────────────
try:
    import websockets
except ImportError:
    print("  Installing websockets …")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "websockets>=12.0"])
    import websockets

# ══════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════
HTTP_PORT   = 8080
WS_PORT     = 8765
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DISPLAY_W   = 640      # OpenCV preview window width
DISPLAY_H   = 360      # OpenCV preview window height

# ══════════════════════════════════════════════════════════════════════
#  SHARED STATE  (written by main thread, read by WS thread)
# ══════════════════════════════════════════════════════════════════════
_data_lock = threading.Lock()
_latest: dict = {
    "gesture":        "NONE",
    "dx":             0.0,
    "dy":             0.0,
    "pinch_distance": None,
    "hand_detected":  False,
    "frame_w":        640,
    "frame_h":        480,
}


def _push(gesture_data: dict, frame_w: int, frame_h: int):
    """Update shared gesture state (called from main thread)."""
    with _data_lock:
        _latest.update({
            "gesture":        gesture_data["gesture"],
            "dx":             float(gesture_data["movement"][0]),
            "dy":             float(gesture_data["movement"][1]),
            "pinch_distance": gesture_data.get("pinch_distance"),
            "hand_detected":  True,
            "frame_w":        frame_w,
            "frame_h":        frame_h,
        })


def _push_no_hand():
    """Mark no hand detected."""
    with _data_lock:
        _latest["hand_detected"] = False
        _latest["gesture"]       = "NONE"
        _latest["dx"]            = 0.0
        _latest["dy"]            = 0.0


# ══════════════════════════════════════════════════════════════════════
#  HTTP SERVER  (serves viewer.html from project dir)
# ══════════════════════════════════════════════════════════════════════

class _QuietHandler(SimpleHTTPRequestHandler):
    """Serves files from PROJECT_DIR without cluttering the console."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PROJECT_DIR, **kwargs)

    def log_message(self, *_args):
        pass            # silence per-request logs


def _start_http():
    server = HTTPServer(("localhost", HTTP_PORT), _QuietHandler)
    server.serve_forever()


# ══════════════════════════════════════════════════════════════════════
#  WEBSOCKET SERVER  (pushes gesture JSON at ~30 fps)
# ══════════════════════════════════════════════════════════════════════

async def _ws_handler(ws):
    try:
        while True:
            with _data_lock:
                payload = json.dumps(_latest)
            await ws.send(payload)
            await asyncio.sleep(1 / 30)
    except Exception:
        pass                # client disconnected — that's fine


async def _ws_main():
    async with websockets.serve(_ws_handler, "localhost", WS_PORT):
        await asyncio.Future()        # run forever


def _start_ws():
    asyncio.run(_ws_main())


# ══════════════════════════════════════════════════════════════════════
#  TRACKING WINDOW HUD
# ══════════════════════════════════════════════════════════════════════

def _draw_tracking_hud(frame, gesture: str, fps_val: float):
    """Overlay HUD on the small tracking preview."""
    h, w = frame.shape[:2]
    accent = (255, 255, 0)      # cyan in BGR
    dim    = (0, 150, 140)

    draw_hud_text(frame, "TRACKING FEED", (10, 24),
                  color=accent, scale=0.55, thickness=2)
    draw_hud_text(frame, f"FPS: {fps_val:.0f}", (10, 46),
                  color=(200, 220, 0), scale=0.42)

    # Gesture label
    g_colors = {
        "NONE":        (100, 130, 120),
        "POINT":       (255, 200, 0),
        "PINCH":       (100, 255, 0),
        "FIST":        (0, 140, 255),
        "OPEN PALM":   (0, 255, 255),
        "SWIPE LEFT":  (255, 0, 255),
        "SWIPE RIGHT": (255, 0, 255),
    }
    draw_hud_text(frame, f"GESTURE: {gesture}", (10, h - 14),
                  color=g_colors.get(gesture, (200, 200, 200)), scale=0.5)

    # Corner brackets
    L = 22
    cv2.line(frame, (4, 4), (4 + L, 4),     dim, 1, cv2.LINE_AA)
    cv2.line(frame, (4, 4), (4, 4 + L),     dim, 1, cv2.LINE_AA)
    cv2.line(frame, (w-5, 4), (w-5-L, 4),   dim, 1, cv2.LINE_AA)
    cv2.line(frame, (w-5, 4), (w-5, 4+L),   dim, 1, cv2.LINE_AA)
    cv2.line(frame, (4, h-5), (4+L, h-5),   dim, 1, cv2.LINE_AA)
    cv2.line(frame, (4, h-5), (4, h-5-L),   dim, 1, cv2.LINE_AA)
    cv2.line(frame, (w-5, h-5), (w-5-L, h-5), dim, 1, cv2.LINE_AA)
    cv2.line(frame, (w-5, h-5), (w-5, h-5-L), dim, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    # ── Banner ────────────────────────────────────────────────────────
    print()
    print("=" * 56)
    print("   GESTURE CONTROLLED 3D — Three.js Mode")
    print("=" * 56)
    print()
    print(f"  HTTP  →  http://localhost:{HTTP_PORT}/viewer.html")
    print(f"  WS    →  ws://localhost:{WS_PORT}")
    print()

    # ── Start background servers ──────────────────────────────────────
    threading.Thread(target=_start_http, daemon=True).start()
    threading.Thread(target=_start_ws,   daemon=True).start()
    time.sleep(0.4)
    print("  ✓ Servers started")

    # ── Open webcam ───────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [ERROR] Could not open webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    ret, first = cap.read()
    if not ret:
        print("  [ERROR] Could not read from webcam!")
        cap.release()
        return
    full_h, full_w = first.shape[:2]
    print(f"  ✓ Webcam ready  ({full_w}×{full_h})")

    # ── Init tracking components ──────────────────────────────────────
    tracker    = HandTracker(max_hands=1, detection_conf=0.7, tracking_conf=0.7)
    recognizer = GestureRecognizer()
    fps_ctr    = FPSCounter()
    trail      = HandTrail(max_length=25)

    print("  ✓ Hand tracker ready")
    print()

    # ── Open browser ──────────────────────────────────────────────────
    url = f"http://localhost:{HTTP_PORT}/viewer.html"
    webbrowser.open(url)
    print(f"  → Browser opened: {url}")
    print("  Press Q in the tracking window to quit.\n")

    # ── Tracking loop ─────────────────────────────────────────────────
    gesture_label = "NONE"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        # ── Process hand tracking (full-res for accuracy) ─────────────
        hand_found = tracker.process(frame)

        # ── Build preview (darkened + tinted) ─────────────────────────
        display = (frame.astype(np.float32) * 0.28).astype(np.uint8)
        tint = np.zeros_like(display)
        tint[:, :, 0] = 18
        tint[:, :, 1] = 6
        display = cv2.add(display, tint)

        if hand_found:
            gesture_data = recognizer.recognize(tracker)
            gesture_label = gesture_data["gesture"]
            _push(gesture_data, fw, fh)

            tracker.draw_landmarks(display, color=(255, 255, 0))
            trail.add(gesture_data.get("hand_center"))
            trail.draw(display, color=(255, 200, 0))

            # Pinch indicator
            if gesture_label == "PINCH":
                thumb = tracker.get_thumb_tip()
                index = tracker.get_index_tip()
                if thumb is not None and index is not None:
                    mid = ((thumb + index) / 2).astype(int)
                    cv2.circle(display, tuple(mid), 10, (100, 255, 0), 2, cv2.LINE_AA)
        else:
            gesture_label = "NONE"
            _push_no_hand()

        # ── HUD + scanlines ───────────────────────────────────────────
        draw_scanline_overlay(display, intensity=0.02)
        fps_ctr.tick()
        _draw_tracking_hud(display, gesture_label, fps_ctr.get_fps())

        # ── Resize & show ─────────────────────────────────────────────
        small = cv2.resize(display, (DISPLAY_W, DISPLAY_H),
                           interpolation=cv2.INTER_AREA)
        cv2.imshow("Hand Tracking — Gesture Control 3D", small)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    # ── Cleanup ───────────────────────────────────────────────────────
    tracker.release()
    cap.release()
    cv2.destroyAllWindows()
    print("\n  Shutdown complete.\n")


if __name__ == "__main__":
    main()
