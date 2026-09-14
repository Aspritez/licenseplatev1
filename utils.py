"""Utility functions for image conversion and drawing."""
import cv2
import numpy as np
import base64

def image_to_base64(img: np.ndarray) -> str:
    """Convert numpy BGR image to base64 PNG string."""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

def base64_to_image(b64_str: str) -> np.ndarray:
    """Convert base64 string to numpy BGR image."""
    img_bytes = base64.b64decode(b64_str)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def bytes_to_image(file_bytes: bytes) -> np.ndarray | None:
    """Convert uploaded file bytes to numpy BGR image. Returns None if invalid."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def draw_corners_on_image(img: np.ndarray, corners: np.ndarray, color=(0, 255, 0), radius=8, thickness=2) -> np.ndarray:
    """Draw 4 corner points and connecting lines on image copy."""
    result = img.copy()
    pts = corners.astype(int)
    # Draw lines connecting corners
    for i in range(4):
        cv2.line(result, tuple(pts[i]), tuple(pts[(i+1) % 4]), color, thickness)
    # Draw corner circles
    for pt in pts:
        cv2.circle(result, tuple(pt), radius, (0, 0, 255), -1)
        cv2.circle(result, tuple(pt), radius, color, thickness)
    return result

def draw_all_candidates(img: np.ndarray, contours: list, best_idx: int) -> np.ndarray:
    """Draw all candidate contours. Best in green, others in yellow."""
    result = img.copy()
    for i, cnt in enumerate(contours):
        if i == best_idx:
            cv2.drawContours(result, [cnt], -1, (0, 255, 0), 3)
        else:
            cv2.drawContours(result, [cnt], -1, (0, 255, 255), 2)
    return result
