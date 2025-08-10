import numpy as np
import cv2
import time
import sys

# Optional: Numba for JIT acceleration of pure-Python/NumPy sections
try:
    from numba import njit, prange  # prange reserved for future parallelization
except Exception:  # Safe fallback when Numba is not installed
    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper

    def prange(*args, **kwargs):
        return range(*args, **kwargs)


@njit(cache=True, fastmath=True)
def _mean_luminance_u8(bgr: np.ndarray) -> float:
    """
    Fast mean luminance for uint8 BGR frames using Rec. 601 coefficients.
    Returns value in [0, 1].
    """
    h, w, _ = bgr.shape
    total = 0.0
    for y in range(h):
        for x in range(w):
            b = bgr[y, x, 0]
            g = bgr[y, x, 1]
            r = bgr[y, x, 2]
            total += 0.114 * b + 0.587 * g + 0.299 * r
    return total / (255.0 * (h * w))


@njit(cache=True)
def _percentiles_bgr_u8(bgr: np.ndarray, percentile: float):
    """
    Compute channel-wise high percentiles via 256-bin histograms for uint8 BGR images.
    Returns (vb, vg, vr) as floats in [0, 255].
    """
    h, w, _ = bgr.shape
    hb = np.zeros(256, np.int64)
    hg = np.zeros(256, np.int64)
    hr = np.zeros(256, np.int64)

    for y in range(h):
        for x in range(w):
            b = bgr[y, x, 0]
            g = bgr[y, x, 1]
            r = bgr[y, x, 2]
            hb[b] += 1
            hg[g] += 1
            hr[r] += 1

    N = h * w
    k = int(np.ceil((percentile / 100.0) * N)) - 1
    if k < 0:
        k = 0

    def from_hist(hist, kth):
        c = 0
        for v in range(256):
            c += hist[v]
            if c > kth:
                return float(v)
        return 255.0

    vb = from_hist(hb, k)
    vg = from_hist(hg, k)
    vr = from_hist(hr, k)
    return vb, vg, vr


def calc_mean(img_bgr):
    """
    Calculate the mean luminance of an image in BGR format.

    This function converts a BGR image (e.g., as loaded by OpenCV) from 
    8-bit integer values [0, 255] to floating-point values [0.0, 1.0],
    extracts the Blue, Green, and Red channels, and computes the mean 
    luminance using the standard Rec. 601 formula:

        luminance = 0.299 * R + 0.587 * G + 0.114 * B

    Args:
        img_bgr (numpy.ndarray): Input image in BGR color format, 
                                 shape (H, W, 3), dtype uint8.

    Returns:
        float: Mean luminance value in the range [0.0, 1.0].
    """
    # Numba-accelerated path for uint8 frames
    if img_bgr.dtype == np.uint8:
        return float(_mean_luminance_u8(img_bgr))

    # Fallback path for non-uint8
    img_float = img_bgr.astype(np.float32) / 255.0
    b = img_float[:, :, 0]
    g = img_float[:, :, 1]
    r = img_float[:, :, 2]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return float(luminance.mean())


def calc_alpha(p0, mean):
    """
    Calculate the alpha parameter based on a base value and image mean.

    This function applies a simple linear adjustment to the base 
    parameter `p0` using a fixed slope `p1` and the given `mean` value.

        alpha = p0 + (p1 * mean)

    Where:
        p1 is a constant slope value (-0.018).

    Args:
        p0 (float): Base alpha value before adjustment.
        mean (float): Mean value (e.g., image luminance) to adjust alpha.

    Returns:
        float: Adjusted alpha value.
    """
    p1 = -0.018
    alpha = p0 + (p1 * mean)
    return alpha


def enhancement(
    img_bgr,
    alpha,
    out_buffer=None,
    prev_scales=None,
    gain_smooth: float = 0.85,
    percentile: float = 99.5,
):
    """
    Apply per-channel contrast enhancement to a BGR image.

    This function computes scaling factors for each channel (B, G, R)
    based on the high-percentile intensity values, then applies a gain 
    to enhance the image. The gain is smoothed over frames if previous 
    scales are provided (useful for video processing to avoid flicker).

    Args:
        img_bgr (numpy.ndarray): Input image in BGR format, dtype uint8.
        alpha (float): Global enhancement strength multiplier.
        out_buffer (numpy.ndarray, optional): Optional preallocated output 
            buffer to store the result. Must match `img_bgr` shape.
        prev_scales (tuple[float, float, float], optional): Previous frame's 
            enhancement scales for (B, G, R) channels to enable smoothing.
        gain_smooth (float, optional): Smoothing factor between 0 and 1 
            for scale blending. Higher means more smoothing. Default is 0.85.
        percentile (float, optional): Percentile for high-intensity reference 
            in each channel (used to calculate scaling). Default is 99.5.

    Returns:
        tuple:
            - out (numpy.ndarray): Enhanced BGR image (uint8).
            - scales (tuple[float, float, float]): Final enhancement scales 
              for (B, G, R) channels.
    """
    # Split channels
    b = img_bgr[:, :, 0]
    g = img_bgr[:, :, 1]
    r = img_bgr[:, :, 2]

    eps = 1e-6

    # Fast percentile path for uint8 frames, else fallback to NumPy
    if img_bgr.dtype == np.uint8:
        vb, vg, vr = _percentiles_bgr_u8(img_bgr, percentile)
    else:
        vb = float(np.percentile(b, percentile))
        vg = float(np.percentile(g, percentile))
        vr = float(np.percentile(r, percentile))

    # Current enhancement factors per channel
    e_r_curr = alpha * (170.7 / (vr + 15.49 + eps))
    e_g_curr = alpha * (179.3 / (vg + 15.42 + eps))
    e_b_curr = alpha * (160.4 / (vb + 15.81 + eps))

    # Smooth with previous scales if present
    if prev_scales is None:
        e_b, e_g, e_r = e_b_curr, e_g_curr, e_r_curr
    else:
        s = float(np.clip(gain_smooth, 0.0, 0.999))
        e_b = s * prev_scales[0] + (1.0 - s) * e_b_curr
        e_g = s * prev_scales[1] + (1.0 - s) * e_g_curr
        e_r = s * prev_scales[2] + (1.0 - s) * e_r_curr

    # Allocate output buffer if needed
    if out_buffer is None or out_buffer.shape != img_bgr.shape:
        out = np.empty_like(img_bgr)
    else:
        out = out_buffer

    # Apply per-channel scaling via OpenCV (highly optimized)
    out[:, :, 0] = cv2.convertScaleAbs(b, alpha=e_b, beta=0)
    out[:, :, 1] = cv2.convertScaleAbs(g, alpha=e_g, beta=0)
    out[:, :, 2] = cv2.convertScaleAbs(r, alpha=e_r, beta=0)

    return out, (e_b, e_g, e_r)


def init_camera(
    camera_index: int,
    width: int,
    height: int,
    target_fps: int,
    lock_auto_exposure: bool,
    manual_exposure: float,
):
    """
    Initialize the webcam with platform-specific backend selection, format fallbacks,
    resolution/FPS setting, buffer size optimization, and optional exposure locking.

    This function attempts multiple capture backends depending on the OS,
    sets MJPEG or YUY2 pixel formats for speed and quality, and ensures
    the camera is opened with the desired parameters.

    Returns:
        tuple: (cap, actual_w, actual_h, actual_fps)
    """
    import sys as _sys

    # Enable OpenCV optimizations
    try:
        cv2.setUseOptimized(True)
    except Exception:
        pass

    # Set OpenCV threads to CPU count
    try:
        nthreads = max(1, cv2.getNumberOfCPUs())
        cv2.setNumThreads(nthreads)
    except Exception:
        pass

    # Try preferred backends
    if _sys.platform == "win32":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
    else:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)

    # Try MJPEG for speed
    if _sys.platform == "win32":
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

    # Set resolution and FPS
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if target_fps > 0:
        cap.set(cv2.CAP_PROP_FPS, target_fps)

    # Reduce buffer for low latency
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    # Non-Windows MJPEG attempt
    if _sys.platform != "win32":
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

    # Fail gracefully if not open
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return None, 0, 0, 0

    # Lock exposure if requested (Windows only)
    if _sys.platform == "win32" and lock_auto_exposure:
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_EXPOSURE, float(manual_exposure))
        except Exception:
            pass

    # Actual parameters
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    # Windows YUY2 fallback if needed
    if _sys.platform == "win32" and (actual_w < width or actual_h < height):
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUY2"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if target_fps > 0:
                cap.set(cv2.CAP_PROP_FPS, target_fps)
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
        except Exception:
            pass

    return cap, actual_w, actual_h, actual_fps


def run_webcam(
    camera_index: int = 1,
    p0: float = 1.6,
    width: int = 1280,
    height: int = 720,
    target_fps: int = 30,
    fast_mean: bool = True,
    mean_downsample: int = 8,
    alpha_update_interval: int = 3,
    alpha_smooth: float = 0.9,
    alpha_min: float = 0.8,
    alpha_max: float = 2.5,
    gain_smooth: float = 0.85,
    gain_percentile: float = 99.5,
    stats_blur: bool = True,
    stats_blur_sigma: float = 1.0,
    lock_auto_exposure: bool = False,
    manual_exposure: float = -5.0,
):
    """
    Capture video frames, enhance them, and display in real-time.

    Uses mean luminance to adjust a global alpha factor and applies 
    percentile-based per-channel scaling for contrast enhancement.
    """

    cap, actual_w, actual_h, actual_fps = init_camera(
        camera_index,
        width,
        height,
        target_fps,
        lock_auto_exposure,
        manual_exposure,
    )
    if cap is None:
        return

    # Warm-up JIT to avoid first-frame latency (safe even without Numba)
    try:
        dummy = np.zeros((16, 16, 3), dtype=np.uint8)
        _ = _mean_luminance_u8(dummy)
        _ = _percentiles_bgr_u8(dummy, gain_percentile)
    except Exception:
        pass

    window_name = "Original | Enhanced (CPU/Numba) - Press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(actual_w * 2, 1920), min(actual_h, 720))

    prev_time = time.time()
    out_buffer = None
    prev_scales = None
    prev_alpha = None
    alpha = 1.0
    frame_count = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None or frame_bgr.size == 0:
            placeholder = np.zeros(
                (height if height > 0 else 480, width if width > 0 else 640, 3), dtype=np.uint8
            )
            cv2.putText(
                placeholder,
                "No frame from camera",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )
            combined_ph = cv2.hconcat([placeholder, placeholder])
            cv2.imshow(window_name, combined_ph)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        # Update alpha periodically
        if frame_count % alpha_update_interval == 0:
            if fast_mean and mean_downsample > 1:
                small = cv2.resize(
                    frame_bgr,
                    (
                        max(1, frame_bgr.shape[1] // mean_downsample),
                        max(1, frame_bgr.shape[0] // mean_downsample),
                    ),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                small = frame_bgr

            if stats_blur and stats_blur_sigma > 0:
                small = cv2.GaussianBlur(small, (0, 0), stats_blur_sigma)

            mean_value = calc_mean(small)
            alpha_raw = calc_alpha(p0, mean_value)
            alpha_raw = float(np.clip(alpha_raw, alpha_min, alpha_max))

            if prev_alpha is None:
                alpha = alpha_raw
            else:
                s = float(np.clip(alpha_smooth, 0.0, 0.999))
                alpha = s * prev_alpha + (1.0 - s) * alpha_raw
            prev_alpha = alpha

        # Apply enhancement
        out_buffer, prev_scales = enhancement(
            frame_bgr,
            alpha,
            out_buffer=out_buffer,
            prev_scales=prev_scales,
            gain_smooth=gain_smooth,
            percentile=gain_percentile,
        )

        # Display FPS and parameters
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        info = f"{actual_w}x{actual_h} @{actual_fps:.0f} alpha={alpha:.3f} fps={fps:.1f}"

        # Prepare side-by-side view: original (left) and enhanced (right)
        orig_vis = frame_bgr.copy()
        out_vis = out_buffer.copy()

        cv2.putText(orig_vis, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(out_vis, "Enhanced", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(out_vis, info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        combined = cv2.hconcat([orig_vis, out_vis])

        cv2.imshow(window_name, combined)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()