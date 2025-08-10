import numpy as np
import cv2
import time


def calc_mean(img_rgb):
    """
    Calculate the scalar mean luminance (normalized 0..1) as 0.299 R + 0.587 G + 0.114 B
    Input: image in RGB

    Output: scalar mean in [0, 1]
    """
    img_float = img_rgb.astype(np.float32) / 255.0
    luminance = 0.299 * img_float[:, :, 0] + 0.587 * img_float[:, :, 1] + 0.114 * img_float[:, :, 2]
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


def enhancement(img_rgb, alpha):
    """
    Final enhancement as shown in equation 10 of the paper
    Input: image in RGB
           alpha

    Output: enhanced image in BGR format
    """
    img_float = img_rgb.astype(np.float32)

    D_r = img_float[:, :, 0]
    D_g = img_float[:, :, 1]
    D_b = img_float[:, :, 2]

    # Avoid division by zero by adding a tiny epsilon
    max_r = float(np.max(D_r))
    max_g = float(np.max(D_g))
    max_b = float(np.max(D_b))
    E_r = alpha * (170.7 / (max_r + 15.49 + 1e-6))
    E_g = alpha * (179.3 / (max_g + 15.42 + 1e-6))
    E_b = alpha * (160.4 / (max_b + 15.81 + 1e-6))

    R = D_r * E_r
    G = D_g * E_g
    B = D_b * E_b

    enhanced_bgr = np.dstack([B, G, R])
    enhanced_bgr = np.clip(enhanced_bgr, 0, 255).astype(np.uint8)
    return enhanced_bgr


def run_webcam(
    camera_index: int = 0,
    p0: float = 1.6,
    width: int = 1920,
    height: int = 1080,
    target_fps: int = 30,
    fast_mean: bool = True,
    mean_downsample: int = 4,
):
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

    # Prefer MJPG for better throughput if available
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

    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None or frame_bgr.size == 0:
            placeholder = np.zeros((height if height > 0 else 480, width if width > 0 else 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "No frame from camera", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow(window_name, placeholder)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if fast_mean and mean_downsample > 1:
            small = cv2.resize(
                frame_rgb,
                (max(1, frame_rgb.shape[1] // mean_downsample), max(1, frame_rgb.shape[0] // mean_downsample)),
                interpolation=cv2.INTER_AREA,
            )
            mean_value = calc_mean(small)
        else:
            mean_value = calc_mean(frame_rgb)

        alpha = calc_alpha(p0, mean_value)
        enhanced_bgr = enhancement(frame_rgb, alpha)

        # Overlay quick diagnostics
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        info = f"{actual_w}x{actual_h} @{actual_fps:.0f} req:{width}x{height}@{target_fps} alpha={alpha:.3f} fps={fps:.1f}"
        cv2.putText(enhanced_bgr, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, enhanced_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()