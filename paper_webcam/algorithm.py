import numpy as np


def compute_luminance_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    """Compute luminance from BGR using Rec.709 luma weights.

    Parameters
    ----------
    frame_bgr: np.ndarray
        HxWx3 uint8 array in BGR order.

    Returns
    -------
    np.ndarray
        HxW float32 array with values in [0, 1].
    """
    if frame_bgr.dtype != np.uint8:
        raise ValueError("frame_bgr must be uint8")
    # Convert to float32 in [0,1]
    b = frame_bgr[..., 0].astype(np.float32) * (1.0 / 255.0)
    g = frame_bgr[..., 1].astype(np.float32) * (1.0 / 255.0)
    r = frame_bgr[..., 2].astype(np.float32) * (1.0 / 255.0)
    # Rec.709 coefficients
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def choose_adaptive_gamma(mean_luminance: float) -> float:
    """Map mean luminance in [0,1] to a gamma value.

    Bright scenes (mean close to 1) -> gamma > 1 (slight darkening)
    Dark scenes (mean close to 0) -> gamma < 1 (brightening)
    """
    # Center around 0.5 and increase contrast away from 0.5.
    # Clamp to a reasonable range for stability.
    gamma = 1.0 + 1.2 * (0.5 - float(mean_luminance))
    return float(np.clip(gamma, 0.5, 2.2))


def build_gamma_lut(gamma: float) -> np.ndarray:
    """Build 256-entry LUT for gamma correction.

    Returns uint8 array so it can be applied with LUT[pixels].
    """
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    y = np.power(x, gamma)
    lut = np.clip(np.round(y * 255.0), 0.0, 255.0).astype(np.uint8)
    return lut


def apply_lut_uint8(frame_uint8: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply 1D uint8 LUT to an uint8 image (HxW or HxWxC).

    This is a pure NumPy operation and very fast.
    """
    if frame_uint8.dtype != np.uint8:
        raise ValueError("frame_uint8 must be uint8")
    if lut.dtype != np.uint8 or lut.shape[0] != 256:
        raise ValueError("lut must be uint8[256]")
    return lut[frame_uint8]


def process_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Paper algorithm entry point (placeholder: adaptive gamma via NumPy only).

    Parameters
    ----------
    frame_bgr: np.ndarray
        HxWx3 uint8 image in BGR order.

    Returns
    -------
    np.ndarray
        HxWx3 uint8 processed image in BGR order.
    """
    # Compute mean luminance quickly
    luminance = compute_luminance_bgr(frame_bgr)
    mean_luma = float(np.mean(luminance))

    # Adaptive gamma selection
    gamma = choose_adaptive_gamma(mean_luma)

    # LUT-based gamma correction
    lut = build_gamma_lut(gamma)
    corrected = apply_lut_uint8(frame_bgr, lut)

    return corrected