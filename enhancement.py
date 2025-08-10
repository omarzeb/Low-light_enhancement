import numpy as np
import cv2


def calc_mean(img_rgb):
    """
    Calculate the scalar mean luminance as defined by 0.299 R + 0.587 G + 0.114 B
    Input: image in RGB

    Output: scalar mean
    """
    img_float = img_rgb.astype(np.float32)
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

    E_r = alpha * (170.7 / (float(np.max(D_r)) + 15.49))
    E_g = alpha * (179.3 / (float(np.max(D_g)) + 15.42))
    E_b = alpha * (160.4 / (float(np.max(D_b)) + 15.81))

    R = D_r * E_r
    G = D_g * E_g
    B = D_b * E_b

    enhanced_bgr = np.dstack([B, G, R])
    enhanced_bgr = np.clip(enhanced_bgr, 0, 255).astype(np.uint8)
    return enhanced_bgr


def run_webcam(camera_index: int = 0, p0: float = 1.6, width: int = 640, height: int = 480):
    cap = cv2.VideoCapture(camera_index)
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    window_name = "Enhanced (CPU) - Press 'q' to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        mean_value = calc_mean(frame_rgb)
        alpha = calc_alpha(p0, mean_value)
        enhanced_bgr = enhancement(frame_rgb, alpha)

        cv2.imshow(window_name, enhanced_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()