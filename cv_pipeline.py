import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Any
from coordinate_transform import order_points, validate_quadrilateral
from utils import bytes_to_image, draw_corners_on_image, draw_all_candidates

@dataclass
class StageResult:
    success: bool
    confidence: float
    data: Any
    message: str
    debug_image: Optional[np.ndarray]

@dataclass
class PipelineResult:
    success: bool
    error_code: Optional[str]
    error_message: Optional[str]
    preprocess_confidence: float
    contour_confidence: float
    corner_confidence: float
    overall_detection_confidence: float
    deblur_applied: bool
    deblur_method: Optional[str]
    blur_score_before: float
    blur_score_after: float
    corners: Optional[np.ndarray]
    annotated_image: Optional[np.ndarray]
    candidate_contours: List[np.ndarray]

def preprocess(img: np.ndarray) -> np.ndarray:
    """Grayscale, Blur, CLAHE."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(blurred)

def detect_and_correct_blur(gray_img: np.ndarray) -> Tuple[np.ndarray, float, float, Optional[str]]:
    """Detect blur and apply correction if needed."""
    blur_score = cv2.Laplacian(gray_img, cv2.CV_64F).var()

    # A near-flat image has no recoverable detail. Running Wiener deconvolution
    # would create ringing edges that could be mistaken for a plate contour.
    if blur_score < 1.0:
        return gray_img, blur_score, blur_score, None
    
    if blur_score >= 200:
        return gray_img, blur_score, blur_score, None
        
    elif 50 <= blur_score < 200:
        # Unsharp masking
        blurred = cv2.GaussianBlur(gray_img, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(gray_img, 1.5, blurred, -0.5, 0)
        new_score = cv2.Laplacian(sharpened, cv2.CV_64F).var()
        return sharpened, blur_score, new_score, "UNSHARP_MASKING"
        
    else:
        # Wiener Deconvolution
        # Create Gaussian PSF
        k_size = 13
        sigma = 5
        psf = cv2.getGaussianKernel(k_size, sigma)
        psf = psf * psf.T
        psf /= psf.sum()
        
        img_float = gray_img.astype(np.float64)
        
        # Pad PSF to image size
        psf_padded = np.zeros_like(img_float)
        psf_padded[:k_size, :k_size] = psf
        # Circular shift
        psf_padded = np.roll(psf_padded, -k_size//2, axis=0)
        psf_padded = np.roll(psf_padded, -k_size//2, axis=1)
        
        # DFT
        img_fft = np.fft.fft2(img_float)
        psf_fft = np.fft.fft2(psf_padded)
        
        # Wiener filter
        K = 0.01
        psf_fft_conj = np.conj(psf_fft)
        H_weiner = psf_fft_conj / (np.abs(psf_fft)**2 + K)
        
        result_fft = img_fft * H_weiner
        result_float = np.real(np.fft.ifft2(result_fft))
        
        # Normalize
        result = cv2.normalize(result_float, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        new_score = cv2.Laplacian(result, cv2.CV_64F).var()
        return result, blur_score, new_score, "WIENER_DECONVOLUTION"

def detect_edges(gray: np.ndarray) -> np.ndarray:
    """Edge detection and dilation."""
    edges = cv2.Canny(gray, 50, 200)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(edges, kernel, iterations=1)

def find_plate_contours(gray: np.ndarray, edges: np.ndarray) -> List[np.ndarray]:
    """Find contours with geometry consistent with a vehicle plate.

    Road texture and vehicle body panels create far more edges than a plate.
    Rejecting border-touching and over-sized contours here prevents the image
    frame, a bumper, or the whole car from reaching the scoring stage.
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    image_h, image_w = gray.shape[:2]
    image_area = image_h * image_w
    min_contour_area = max(500.0, image_area * 0.0001)
    max_box_area = image_area * 0.025

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_contour_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if h == 0:
            continue

        # Contours on the image frame are not plate candidates.
        margin = 3
        if x <= margin or y <= margin or x + w >= image_w - margin or y + h >= image_h - margin:
            continue

        box_area = w * h
        aspect_ratio = float(w) / h
        rectangularity = area / box_area
        if not (1.2 <= aspect_ratio <= 4.5):
            continue
        if box_area > max_box_area or rectangularity < 0.15:
            continue

        candidates.append(cnt)

    return candidates

def select_best_contour(
    candidates: List[np.ndarray], img_area: float, image_shape: Tuple[int, int], gray: Optional[np.ndarray] = None
) -> Tuple[Optional[np.ndarray], int, float]:
    """Score candidates by plate-like geometry and expected object location."""
    if not candidates:
        return None, -1, 0.0
        
    best_score = -1
    best_cnt = None
    best_idx = -1
    
    image_h, image_w = image_shape
    for i, cnt in enumerate(candidates):
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = float(w) / h

        # Thai plates are usually close to 2.2:1, but perspective can make a
        # valid plate appear narrower or wider. Log distance keeps the score
        # symmetric around the target ratio.
        aspect_score = np.exp(-abs(np.log(aspect / 2.2)) / 0.8)

        # A plate is a small but visible part of these full-vehicle photos.
        # Score its bounding-box proportion around 0.3% of the whole image.
        box_fraction = (w * h) / img_area
        area_score = np.exp(-abs(np.log(box_fraction / 0.003)) / 0.9)

        rectangularity = area / (w * h)
        rectangularity_score = float(np.clip(rectangularity, 0.0, 1.0))

        # Rear plates are generally near the horizontal centre and in the
        # lower half of a vehicle image.  The previous upper-middle bias often
        # preferred foliage or a bumper detail over the actual plate.
        centre_x = (x + w / 2) / image_w
        centre_y = (y + h / 2) / image_h
        location_score = np.exp(
            -(
                ((centre_x - 0.5) ** 2) / (2 * 0.24 ** 2)
                + ((centre_y - 0.60) ** 2) / (2 * 0.23 ** 2)
            )
        )

        # A Thai registration plate is usually noticeably brighter than the
        # surrounding car body, road, or vegetation.  This is a soft score so
        # it still accepts shadowed or dirty plates.
        brightness_score = 0.5
        if gray is not None:
            roi = gray[y:y + h, x:x + w]
            if roi.size:
                brightness_score = float(np.clip((roi.mean() - 55.0) / 145.0, 0.0, 1.0))

        score = (
            0.25 * aspect_score
            + 0.15 * area_score
            + 0.15 * rectangularity_score
            + 0.25 * location_score
            + 0.20 * brightness_score
        )
        if score > best_score:
            best_score = score
            best_cnt = cnt
            best_idx = i
            
    confidence = min(100.0, best_score * 100)
    return best_cnt, best_idx, confidence

def detect_corners(contour: np.ndarray, gray: np.ndarray) -> Tuple[Optional[np.ndarray], float]:
    """Extract 4 corners from contour."""
    peri = cv2.arcLength(contour, True)
    
    # Try different epsilons to find 4 corners
    for eps_factor in np.linspace(0.01, 0.05, 10):
        approx = cv2.approxPolyDP(contour, eps_factor * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
            # Refine
            cv2.cornerSubPix(gray, pts, (20, 20), (-1, -1),
                             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            return pts, 80.0
            
    # Fallback to minAreaRect
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    # np.int0 was removed in NumPy 2.0. Keep the sub-pixel coordinates that
    # OpenCV returns; perspective transformation accepts float32 directly.
    return box.astype(np.float32), 40.0


def expand_two_row_plate_corners(corners: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """Expand a detected text-row contour to include a Thai plate's lower row.

    Strong characters often form a cleaner contour than the white plate border.
    If that contour is unusually wide and short, it is likely the registration
    row alone. Thai plates commonly place the province below it, so extend the
    lower edge proportionally while keeping normal-height plate contours intact.
    """
    ordered = order_points(corners.astype(np.float32))
    top_width = np.linalg.norm(ordered[1] - ordered[0])
    bottom_width = np.linalg.norm(ordered[2] - ordered[3])
    left_height = np.linalg.norm(ordered[3] - ordered[0])
    right_height = np.linalg.norm(ordered[2] - ordered[1])
    average_width = (top_width + bottom_width) / 2
    average_height = max((left_height + right_height) / 2, 1.0)
    aspect_ratio = average_width / average_height

    # A normal full plate is close to 2.2:1. Only extend contours that are
    # distinctly wider than that, avoiding unnecessary padding on valid plates.
    if aspect_ratio <= 2.6:
        return ordered

    extra_height_ratio = min(0.65, max(0.0, aspect_ratio / 2.15 - 1.0))
    top_padding = extra_height_ratio * 0.18
    bottom_padding = extra_height_ratio * 0.82
    horizontal_padding = 0.03

    top_edge = ordered[1] - ordered[0]
    bottom_edge = ordered[2] - ordered[3]
    left_edge = ordered[3] - ordered[0]
    right_edge = ordered[2] - ordered[1]
    expanded = np.array([
        ordered[0] - horizontal_padding * top_edge - top_padding * left_edge,
        ordered[1] + horizontal_padding * top_edge - top_padding * right_edge,
        ordered[2] + horizontal_padding * bottom_edge + bottom_padding * right_edge,
        ordered[3] - horizontal_padding * bottom_edge + bottom_padding * left_edge,
    ], dtype=np.float32)

    image_h, image_w = image_shape
    expanded[:, 0] = np.clip(expanded[:, 0], 0, image_w - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, image_h - 1)
    return expanded


def blur_score_to_confidence(blur_score: float) -> float:
    """Map final Laplacian variance to a deterministic 0–100 score.

    A variance of 200 or above is considered sharp by the documented pipeline,
    so it maps to 100. Values below that threshold scale linearly.
    """
    return float(np.clip((blur_score / 200.0) * 100.0, 0.0, 100.0))

def run_detection(image_bytes: bytes) -> PipelineResult:
    """Full CV pipeline."""
    img = bytes_to_image(image_bytes)
    if img is None:
        return PipelineResult(False, "INVALID_IMAGE", "Could not decode image", 
                              0, 0, 0, 0, False, None, 0, 0, None, None, [])
                              
    img_h, img_w = img.shape[:2]
    img_area = img_h * img_w
    
    # Preprocess
    gray = preprocess(img)

    # Deblur
    deblurred, blur_before, blur_after, deblur_method = detect_and_correct_blur(gray)
    deblur_applied = deblur_method is not None
    prep_conf = blur_score_to_confidence(blur_after)
    
    # Edges & Contours
    edges = detect_edges(deblurred)
    candidates = find_plate_contours(deblurred, edges)
    
    if not candidates:
        return PipelineResult(False, "NO_PLATE", "ไม่พบป้ายทะเบียนที่ตรวจจับได้ในภาพ",
                              prep_conf, 0.0, 0.0, 0.0, deblur_applied, deblur_method,
                              blur_before, blur_after, None, img, [])
                              
    best_cnt, best_idx, cont_conf = select_best_contour(candidates, img_area, (img_h, img_w), deblurred)
    
    # Corners
    corners, corn_conf = detect_corners(best_cnt, deblurred)
    
    if corners is None:
        return PipelineResult(False, "NO_CORNERS", "ไม่สามารถระบุมุมป้ายทะเบียนได้",
                              prep_conf, cont_conf, 0.0, 0.0, deblur_applied, deblur_method,
                              blur_before, blur_after, None, img, candidates)

    corners = expand_two_row_plate_corners(corners, (img_h, img_w))
    corners = order_points(corners)
    corners_valid, _ = validate_quadrilateral(corners)
    if not corners_valid:
        return PipelineResult(False, "NO_CORNERS", "ไม่สามารถระบุมุมป้ายทะเบียนที่ไม่ซ้ำกันได้",
                              prep_conf, cont_conf, 0.0, 0.0, deblur_applied, deblur_method,
                              blur_before, blur_after, None, img, candidates)
                               
    overall = 0.2 * prep_conf + 0.4 * cont_conf + 0.4 * corn_conf
    
    # Drawing every low-level edge contour makes the diagnostic image unreadable
    # and does not help a user correct the selected plate. Show only the chosen
    # contour and its corners instead.
    annotated = draw_all_candidates(img, [best_cnt], 0)
    annotated = draw_corners_on_image(annotated, corners)
    
    return PipelineResult(True, None, None, prep_conf, cont_conf, corn_conf, overall,
                          deblur_applied, deblur_method, blur_before, blur_after,
                          corners, annotated, candidates)
