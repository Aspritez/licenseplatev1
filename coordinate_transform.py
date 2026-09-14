import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class TransformResult:
    success: bool
    error_code: Optional[str]
    error_message: Optional[str]
    warped_image: Optional[np.ndarray]
    ordered_corners: Optional[np.ndarray]
    transform_matrix: Optional[np.ndarray]

def order_points(pts: np.ndarray) -> np.ndarray:
    """Sort four distinct points into TL, TR, BR, BL order.

    The common sum/difference shortcut can assign one source point to two
    corners when a plate is strongly rotated or two sums are tied.  Ordering
    around the centroid keeps each input point exactly once.
    """
    points = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    centre = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    ordered = points[np.argsort(angles)]

    # In image coordinates the angular order is TL, TR, BR, BL after rotating
    # the sequence so that its visually top-left point is first.
    top_left_index = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
    return np.roll(ordered, -top_left_index, axis=0).astype(np.float32)


def validate_distinct_points(pts: np.ndarray, minimum_distance: float = 5.0) -> Tuple[bool, str]:
    """Reject repeated or nearly repeated source points before ordering."""
    points = np.asarray(pts, dtype=np.float32)
    if points.shape != (4, 2):
        return False, "ต้องระบุจุดมุม 4 จุด โดยแต่ละจุดมีพิกัด x และ y"
    if not np.isfinite(points).all():
        return False, "พิกัดจุดมุมไม่ถูกต้อง"

    for i in range(4):
        for j in range(i + 1, 4):
            if np.linalg.norm(points[i] - points[j]) < minimum_distance:
                return False, "จุดมุมทั้ง 4 ต้องไม่ซ้ำกันหรืออยู่ใกล้กันเกินไป"
    return True, ""

def validate_quadrilateral(pts: np.ndarray) -> Tuple[bool, str]:
    """Validate four ordered points as a non-degenerate convex quadrilateral."""
    pts = np.asarray(pts, dtype=np.float32)
    is_distinct, error_message = validate_distinct_points(pts)
    if not is_distinct:
        return False, error_message

    def orientation(a, b, c):
        return float(np.cross(b - a, c - a))

    def segments_intersect(a, b, c, d):
        ab_c = orientation(a, b, c)
        ab_d = orientation(a, b, d)
        cd_a = orientation(c, d, a)
        cd_b = orientation(c, d, b)
        return ab_c * ab_d < 0 and cd_a * cd_b < 0

    if segments_intersect(pts[0], pts[1], pts[2], pts[3]) or segments_intersect(pts[1], pts[2], pts[3], pts[0]):
        return False, "เส้นขอบของรูปสี่เหลี่ยมตัดกัน"

    turns = np.array([orientation(pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]) for i in range(4)])
    if np.any(np.isclose(turns, 0.0)) or not (np.all(turns > 0) or np.all(turns < 0)):
        return False, "จุดมุมต้องสร้างรูปสี่เหลี่ยมนูนที่ไม่บิดงอ"

    area = abs(float(cv2.contourArea(pts)))
    if area < 100.0:
        return False, "พื้นที่ของป้ายทะเบียนเล็กเกินไป"
    return True, ""

def compute_perspective_matrix(src_pts: np.ndarray, dst_size: Tuple[int, int] = (400, 200)) -> np.ndarray:
    """Compute perspective transform matrix."""
    dst_pts = np.array([
        [0, 0],
        [dst_size[0] - 1, 0],
        [dst_size[0] - 1, dst_size[1] - 1],
        [0, dst_size[1] - 1]
    ], dtype="float32")
    return cv2.getPerspectiveTransform(src_pts, dst_pts)

def compute_homography_ransac(src_pts: np.ndarray, dst_pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute homography matrix with RANSAC."""
    return cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

def warp_perspective(img: np.ndarray, M: np.ndarray, dst_size: Tuple[int, int] = (400, 200)) -> np.ndarray:
    """Warp image perspective."""
    return cv2.warpPerspective(img, M, dst_size)

def transform_plate(
    img: np.ndarray,
    corners: np.ndarray,
    dst_size: Tuple[int, int] = (400, 200),
    validate_input_order: bool = False,
) -> TransformResult:
    """Combined transform pipeline."""
    try:
        pts = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        if not np.isfinite(pts).all():
            return TransformResult(False, "INVALID_POINTS", "พิกัดจุดมุมไม่ถูกต้อง", None, None, None)

        height, width = img.shape[:2]
        if (pts[:, 0] < 0).any() or (pts[:, 0] >= width).any() or (pts[:, 1] < 0).any() or (pts[:, 1] >= height).any():
            return TransformResult(False, "INVALID_POINTS", "จุดมุมต้องอยู่ภายในขอบเขตของรูปภาพ", None, None, None)

        is_distinct, err_msg = validate_distinct_points(pts)
        if not is_distinct:
            return TransformResult(False, "INVALID_POINTS", err_msg, None, None, None)

        if validate_input_order:
            is_valid, err_msg = validate_quadrilateral(pts)
            if not is_valid:
                return TransformResult(False, "INVALID_POINTS", err_msg, None, None, None)

        ordered_pts = order_points(pts)
        
        is_valid, err_msg = validate_quadrilateral(ordered_pts)
        if not is_valid:
            return TransformResult(False, "INVALID_POINTS", err_msg, None, None, None)
            
        M = compute_perspective_matrix(ordered_pts, dst_size)
        warped = warp_perspective(img, M, dst_size)
        
        return TransformResult(True, None, None, warped, ordered_pts, M)
        
    except (ValueError, cv2.error):
        return TransformResult(False, "DEGENERATE_QUAD", "ไม่สามารถสร้างภาพป้ายทะเบียนจากจุดมุมที่ระบุได้", None, None, None)
