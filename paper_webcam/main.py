import argparse
import os
import sys
import time
from typing import Tuple, Optional

import cv2
import numpy as np

from . import algorithm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam processing (NumPy on CPU)")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--scale", type=float, default=1.0, help="Optional downscale factor for processing")
    parser.add_argument("--use-dshow", type=int, default=0, help="On Windows, use DirectShow backend (1/0)")
    parser.add_argument("--overlay", type=int, default=1, help="Show FPS overlay (1/0)")
    parser.add_argument("--gray", type=int, default=0, help="Toggle grayscale output (1/0)")
    return parser.parse_args()


def open_camera(index: int, width: int, height: int, use_dshow: bool) -> cv2.VideoCapture:
    backend = cv2.CAP_DSHOW if (use_dshow and os.name == "nt") else 0
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    # Lower latency hint
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def maybe_downscale(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return frame
    h, w = frame.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def draw_overlay(frame: np.ndarray, fps: float, gamma: Optional[float] = None) -> None:
    text = f"FPS: {fps:5.1f}"
    if gamma is not None:
        text += f"  gamma: {gamma:0.2f}"
    cv2.putText(
        frame,
        text,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main() -> int:
    args = parse_args()

    cap = open_camera(args.camera_index, args.width, args.height, bool(args.use_dshow))
    if not cap.isOpened():
        print("ERROR: Failed to open camera. Try a different index or backend.", file=sys.stderr)
        return 1

    window_name = "Paper Webcam (NumPy)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_time = time.perf_counter()
    fps = 0.0
    show_overlay = bool(args.overlay)
    show_gray = bool(args.gray)

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("WARN: Failed to read frame")
                break

            if args.scale != 1.0:
                frame_bgr = maybe_downscale(frame_bgr, args.scale)

            # Run paper algorithm (NumPy-only inside)
            processed_bgr = algorithm.process_frame(frame_bgr)

            # Optional grayscale view for debugging
            if show_gray:
                gray = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2GRAY)
                display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            else:
                display = processed_bgr

            # FPS calculation
            now = time.perf_counter()
            dt = now - last_time
            last_time = now
            if dt > 0:
                # Simple low-pass filter for FPS display stability
                current_fps = 1.0 / dt
                fps = 0.9 * fps + 0.1 * current_fps if fps > 0 else current_fps

            if show_overlay:
                # Attempt to introspect gamma from algorithm (if exposed)
                gamma = None
                # If algorithm exposes last gamma, you could retrieve it. Placeholder uses local only.
                draw_overlay(display, fps, gamma)

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('o'):
                show_overlay = not show_overlay
            elif key == ord('g'):
                show_gray = not show_gray

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())