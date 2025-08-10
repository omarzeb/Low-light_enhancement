import numpy as np
import cv2
import time


def calc_mean(img_bgr):
    """
    Calculate the scalar mean luminance (normalized 0..1) using BGR input.
    L = 0.299 * R + 0.587 * G + 0.114 * B

    Input: image in BGR (uint8)
    Output: scalar mean in [0, 1]
    """
    img_float = img_bgr.astype(np.float32) / 255.0
    B = img_float[:, :, 0]
    G = img_float[:, :, 1]
    R = img_float[:, :, 2]
    luminance = 0.299 * R + 0.587 * G + 0.114 * B
    return float(luminance.mean())


def calc_alpha(p0, mean):
    """
    Calculate the value of alpha as given in equation 9 of the paper
    Input: p0 (default to 1.6)
           mean

    Output: alpha
    """
    p1 = -0.018
    alpha = p0 + (p1 * mean)
    return alpha


def enhancement(img_bgr, alpha, out_buffer=None):
    """
    Enhancement using BGR input/output.
    Uses OpenCV's convertScaleAbs for fast per-channel scaling with saturation.
    """
    B = img_bgr[:, :, 0]
    G = img_bgr[:, :, 1]
    R = img_bgr[:, :, 2]

    max_b = float(B.max())
    max_g = float(G.max())
    max_r = float(R.max())

    E_r = alpha * (170.7 / (max_r + 15.49 + 1e-6))
    E_g = alpha * (179.3 / (max_g + 15.42 + 1e-6))
    E_b = alpha * (160.4 / (max_b + 15.81 + 1e-6))

    if out_buffer is None or out_buffer.shape != img_bgr.shape:
        out = np.empty_like(img_bgr)
    else:
        out = out_buffer

    out[:, :, 0] = cv2.convertScaleAbs(B, alpha=E_b, beta=0)
    out[:, :, 1] = cv2.convertScaleAbs(G, alpha=E_g, beta=0)
    out[:, :, 2] = cv2.convertScaleAbs(R, alpha=E_r, beta=0)

    return out


def run_webcam(
    camera_index: int = 0,
    p0: float = 1.6,
    width: int = 1920,
    height: int = 1080,
    target_fps: int = 30,
    fast_mean: bool = True,
    mean_downsample: int = 8,
    alpha_update_interval: int = 3,
):
    # Enable OpenCV optimizations and threads if available
    try:
        cv2.setUseOptimized(True)
    except Exception:
        pass
    try:
        nthreads = max(1, cv2.getNumberOfCPUs())
        cv2.setNumThreads(nthreads)
    except Exception:
        pass

    # Try V4L2 backend first on Linux, then fallback
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    # Set capture properties
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if target_fps > 0:
        cap.set(cv2.CAP_PROP_FPS, target_fps)

    # Reduce latency if supported
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # Try formats: MJPG first, then YUYV
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    except Exception:
        pass

    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    # Read back actual settings
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    window_name = "Enhanced (CPU) - Press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(actual_w, 1280), min(actual_h, 720))

    prev_time = time.time()
    out_buffer = None
    alpha = 1.0
    frame_count = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None or frame_bgr.size == 0:
            placeholder = np.zeros((height if height > 0 else 480, width if width > 0 else 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No frame from camera", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow(window_name, placeholder)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Compute alpha at a lower cadence to reduce CPU
        if frame_count % alpha_update_interval == 0:
            if fast_mean and mean_downsample > 1:
                small = cv2.resize(
                    frame_bgr,
                    (max(1, frame_bgr.shape[1] // mean_downsample), max(1, frame_bgr.shape[0] // mean_downsample)),
                    interpolation=cv2.INTER_AREA,
                )
                mean_value = calc_mean(small)
            else:
                mean_value = calc_mean(frame_bgr)
            alpha = calc_alpha(p0, mean_value)

        out_buffer = enhancement(frame_bgr, alpha, out_buffer)

        # Overlay quick diagnostics
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        info = f"{actual_w}x{actual_h} @{actual_fps:.0f} req:{width}x{height}@{target_fps} alpha={alpha:.3f} fps={fps:.1f}"
        cv2.putText(out_buffer, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, out_buffer)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()