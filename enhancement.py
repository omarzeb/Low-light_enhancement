import numpy as np
import cv2
import time
import sys


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


def enhancement(img_bgr, alpha, out_buffer=None, prev_scales=None, gain_smooth: float = 0.85, percentile: float = 99.5):
    """
    Enhancement using robust percentiles and temporally smoothed per-channel gains.
    Input: BGR frame, scalar alpha, optional output buffer and previous scales.
    Returns: enhanced frame and current smoothed scales as a tuple (E_b, E_g, E_r).
    """
    B = img_bgr[:, :, 0]
    G = img_bgr[:, :, 1]
    R = img_bgr[:, :, 2]

    eps = 1e-6
    vb = float(np.percentile(B, percentile))
    vg = float(np.percentile(G, percentile))
    vr = float(np.percentile(R, percentile))

    E_r_curr = alpha * (170.7 / (vr + 15.49 + eps))
    E_g_curr = alpha * (179.3 / (vg + 15.42 + eps))
    E_b_curr = alpha * (160.4 / (vb + 15.81 + eps))

    if prev_scales is None:
        E_b, E_g, E_r = E_b_curr, E_g_curr, E_r_curr
    else:
        s = float(np.clip(gain_smooth, 0.0, 0.999))
        E_b = s * prev_scales[0] + (1.0 - s) * E_b_curr
        E_g = s * prev_scales[1] + (1.0 - s) * E_g_curr
        E_r = s * prev_scales[2] + (1.0 - s) * E_r_curr

    if out_buffer is None or out_buffer.shape != img_bgr.shape:
        out = np.empty_like(img_bgr)
    else:
        out = out_buffer

    out[:, :, 0] = cv2.convertScaleAbs(B, alpha=E_b, beta=0)
    out[:, :, 1] = cv2.convertScaleAbs(G, alpha=E_g, beta=0)
    out[:, :, 2] = cv2.convertScaleAbs(R, alpha=E_r, beta=0)

    return out, (E_b, E_g, E_r)


# --- Multi-Scale Retinex with Color Restoration (MSRCR) ---

def _simple_color_balance_per_channel(channel: np.ndarray, low_clip: float, high_clip: float) -> np.ndarray:
    """
    Apply simple color balance by clipping low/high percentiles and rescaling to [0, 255].
    Expects float32 input channel. Returns uint8 channel.
    """
    if low_clip <= 0.0 and high_clip <= 0.0:
        ch_min = channel.min()
        ch_max = channel.max()
        if ch_max <= ch_min:
            return np.zeros_like(channel, dtype=np.uint8)
        scaled = (channel - ch_min) * (255.0 / (ch_max - ch_min))
        return np.clip(scaled, 0, 255).astype(np.uint8)

    low_p = 100.0 * float(np.clip(low_clip, 0.0, 0.49))
    high_p = 100.0 * (1.0 - float(np.clip(high_clip, 0.0, 0.49)))
    lo = float(np.percentile(channel, low_p))
    hi = float(np.percentile(channel, high_p))
    if hi <= lo:
        return np.clip(channel, 0, 255).astype(np.uint8)
    clipped = np.clip(channel, lo, hi)
    scaled = (clipped - lo) * (255.0 / (hi - lo))
    return np.clip(scaled, 0, 255).astype(np.uint8)


def msrcr(img_bgr: np.ndarray,
          sigmas=(15.0, 80.0, 250.0),
          weights=None,
          alpha: float = 125.0,
          beta: float = 46.0,
          gain: float = 1.0,
          offset: float = 0.0,
          low_clip: float = 0.01,
          high_clip: float = 0.01) -> np.ndarray:
    """
    Multi-Scale Retinex with Color Restoration (MSRCR) for BGR uint8 images.

    Args:
        img_bgr: Input image, uint8 BGR.
        sigmas: Sequence of Gaussian sigmas for multi-scale Retinex.
        weights: Optional list of weights matching sigmas. Defaults to equal.
        alpha: Color restoration parameter (operates on 0..255 scale).
        beta: Color restoration multiplier.
        gain: Global gain applied to MSRCR output.
        offset: Global offset applied to MSRCR output.
        low_clip, high_clip: Percentile clipping for simple color balance per channel.

    Returns:
        uint8 BGR image enhanced with MSRCR.
    """
    if img_bgr.dtype != np.uint8:
        img = img_bgr.astype(np.float32)
    else:
        img = img_bgr.astype(np.float32)

    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]

    if weights is None:
        weights = [1.0 / max(1, len(sigmas))] * len(sigmas)
    else:
        wsum = float(sum(weights))
        if wsum <= 0:
            weights = [1.0 / max(1, len(sigmas))] * len(sigmas)
        else:
            weights = [w / wsum for w in weights]

    eps = 1e-6
    # Work in 0..255 domain with +1 to avoid log(0)
    Bp = B + 1.0
    Gp = G + 1.0
    Rp = R + 1.0

    # Multi-scale Retinex per channel
    def retinex_multi_scale(channel_plus: np.ndarray) -> np.ndarray:
        ret = np.zeros_like(channel_plus, dtype=np.float32)
        for w, s in zip(weights, sigmas):
            blurred = cv2.GaussianBlur(channel_plus, (0, 0), s) + eps
            ret += w * (np.log(channel_plus) - np.log(blurred))
        return ret

    Rr = retinex_multi_scale(Rp)
    Gr = retinex_multi_scale(Gp)
    Br = retinex_multi_scale(Bp)

    # Color restoration
    sum_rgb = Rp + Gp + Bp
    Rc = beta * (np.log(alpha * Rp) - np.log(sum_rgb))
    Gc = beta * (np.log(alpha * Gp) - np.log(sum_rgb))
    Bc = beta * (np.log(alpha * Bp) - np.log(sum_rgb))

    R_msrcr = gain * (Rc * Rr) + offset
    G_msrcr = gain * (Gc * Gr) + offset
    B_msrcr = gain * (Bc * Br) + offset

    # Simple color balance and normalization per channel to uint8
    out_b = _simple_color_balance_per_channel(B_msrcr, low_clip, high_clip)
    out_g = _simple_color_balance_per_channel(G_msrcr, low_clip, high_clip)
    out_r = _simple_color_balance_per_channel(R_msrcr, low_clip, high_clip)

    out = np.dstack([out_b, out_g, out_r])
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
    alpha_smooth: float = 0.9,
    alpha_min: float = 0.8,
    alpha_max: float = 2.5,
    gain_smooth: float = 0.85,
    gain_percentile: float = 99.5,
    stats_blur: bool = True,
    stats_blur_sigma: float = 1.0,
    lock_auto_exposure: bool = False,
    manual_exposure: float = -5.0,
    algorithm: str = "msrcr",
    retinex_sigmas=(15.0, 80.0, 250.0),
    retinex_weights=None,
    retinex_alpha: float = 125.0,
    retinex_beta: float = 46.0,
    retinex_gain: float = 1.0,
    retinex_offset: float = 0.0,
    retinex_low_clip: float = 0.01,
    retinex_high_clip: float = 0.01,
):
    try:
        cv2.setUseOptimized(True)
    except Exception:
        pass
    try:
        nthreads = max(1, cv2.getNumberOfCPUs())
        cv2.setNumThreads(nthreads)
    except Exception:
        pass

    if sys.platform == "win32":
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)
    else:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)

    if sys.platform == "win32":
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if target_fps > 0:
        cap.set(cv2.CAP_PROP_FPS, target_fps)

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    if sys.platform != "win32":
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    if sys.platform == "win32" and lock_auto_exposure:
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_EXPOSURE, float(manual_exposure))
        except Exception:
            pass

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    if sys.platform == "win32" and (actual_w < width or actual_h < height):
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

    if algorithm.lower() == "msrcr":
        window_name = "Enhanced (MSRCR) - Press 'q' to quit"
    else:
        window_name = "Enhanced (Percentile) - Press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(actual_w, 1280), min(actual_h, 720))

    prev_time = time.time()
    out_buffer = None
    prev_scales = None
    prev_alpha = None
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

        if algorithm.lower() != "msrcr":
            if frame_count % alpha_update_interval == 0:
                if fast_mean and mean_downsample > 1:
                    small = cv2.resize(
                        frame_bgr,
                        (max(1, frame_bgr.shape[1] // mean_downsample), max(1, frame_bgr.shape[0] // mean_downsample)),
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

        if algorithm.lower() == "msrcr":
            out_buffer = msrcr(
                frame_bgr,
                sigmas=retinex_sigmas,
                weights=retinex_weights,
                alpha=retinex_alpha,
                beta=retinex_beta,
                gain=retinex_gain,
                offset=retinex_offset,
                low_clip=retinex_low_clip,
                high_clip=retinex_high_clip,
            )
        else:
            out_buffer, prev_scales = enhancement(
                frame_bgr,
                alpha,
                out_buffer=out_buffer,
                prev_scales=prev_scales,
                gain_smooth=gain_smooth,
                percentile=gain_percentile,
            )

        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        info = f"{actual_w}x{actual_h} @{actual_fps:.0f} alg={algorithm.upper()} fps={fps:.1f}"
        cv2.putText(out_buffer, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, out_buffer)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()