"""Classical plate candidate scoring and contour-based corner extraction.

Scores are heuristics in [0, 1], not calibrated detection probabilities.
No enclosing rectangle is ever substituted for observed plate corners.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


REFERENCE_IMAGE_AREA = 1920 * 1080
FEATURE_WEIGHTS = {
    "area": 0.10,
    "aspect_ratio": 0.15,
    "rectangularity": 0.15,
    "solidity": 0.10,
    "approximation": 0.15,
    "edge_density": 0.35,
}
MIN_CANDIDATE_SCORE = 0.62
MIN_EDGE_DENSITY = 0.012
MAX_REFINEMENT_CANDIDATES = 40


@dataclass(frozen=True)
class DetectorConfig:
    """Explicit scoring parameters for reproducible dataset experiments."""

    weights: tuple[float, ...] = tuple(FEATURE_WEIGHTS.values())
    area_band: tuple[float, float, float, float] = (0.0, 0.003, 0.35, 0.85)
    aspect_band: tuple[float, float, float, float] = (0.8, 1.4, 2.5, 4.5)
    min_aspect: float = 0.9
    max_aspect: float = 4.5
    max_area_fraction: float = 0.85
    minimum_score: float = MIN_CANDIDATE_SCORE
    minimum_edge_density: float = MIN_EDGE_DENSITY
    corner_area_fit: float = 0.88
    direct_rectangularity: float = 0.70
    direct_solidity: float = 0.85


DEFAULT_CONFIG = DetectorConfig()
# Selected on grouped validation data, then frozen before the held-out test.
# Test localization improved only slightly while incorrect returned boxes rose.
# The app uses this for its default dataset preset; DEFAULT_CONFIG above remains
# the reference configuration used by the original near/medium/far presets.
DATASET_CONFIG = DetectorConfig(
    max_area_fraction=1.0,
    min_aspect=0.6,
    aspect_band=(0.5, 1.0, 2.5, 4.5),
    area_band=(0.0, 0.003, 0.98, 1.01),
    corner_area_fit=0.80,
    direct_rectangularity=0.65,
    direct_solidity=0.80,
)


@dataclass
class PlateCandidate:
    contour: np.ndarray
    bbox: tuple[int, int, int, int]
    features: dict[str, float]
    scores: dict[str, float]
    score: float
    corners: np.ndarray | None = None
    corner_source: str | None = None


@dataclass
class DetectionResult:
    candidates: list[PlateCandidate] = field(default_factory=list)
    selected: PlateCandidate | None = None

    @property
    def corners(self) -> np.ndarray | None:
        return None if self.selected is None else self.selected.corners


def order_points(points: np.ndarray) -> np.ndarray:
    """Return distinct source points in TL, TR, BR, BL order."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    centre = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    ordered = points[np.argsort(angles)]
    return np.roll(ordered, -int(np.argmin(ordered[:, 0] + ordered[:, 1])), axis=0).astype(np.float32)


def validate_quadrilateral(points: np.ndarray) -> tuple[bool, str]:
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all():
        return False, "พิกัดจุดมุมไม่ถูกต้อง"
    for first in range(4):
        for second in range(first + 1, 4):
            if np.linalg.norm(points[first] - points[second]) < 8.0:
                return False, "จุดมุมทั้ง 4 ต้องไม่ซ้ำกันหรืออยู่ใกล้กันเกินไป"
    if abs(float(cv2.contourArea(points))) < 100.0:
        return False, "พื้นที่ที่เลือกเล็กเกินไป"
    if not cv2.isContourConvex(points.reshape(-1, 1, 2)):
        return False, "จุดทั้ง 4 ต้องสร้างรูปสี่เหลี่ยมนูนที่ไม่ตัดกัน"
    return True, ""


def _band_score(value: float, low: float, ideal_low: float, ideal_high: float, high: float) -> float:
    return float(np.clip(min((value - low) / (ideal_low - low),
                             (high - value) / (high - ideal_high)), 0.0, 1.0))


def _four_corners(contour: np.ndarray, config: DetectorConfig = DEFAULT_CONFIG) -> np.ndarray | None:
    perimeter = cv2.arcLength(contour, True)
    area = abs(cv2.contourArea(contour))
    for epsilon in (0.01, 0.02, 0.03, 0.04, 0.05):
        approximation = cv2.approxPolyDP(contour, epsilon * perimeter, True)
        if len(approximation) != 4:
            continue
        corners = order_points(approximation)
        quad_area = abs(cv2.contourArea(corners))
        # Reject a forced four-vertex fit to a very different shape.
        if validate_quadrilateral(corners)[0] and min(area, quad_area) / max(area, quad_area, 1) >= config.corner_area_fit:
            return corners
    return None


def score_contour(contour: np.ndarray, text_edges: np.ndarray, minimum_area: float,
                  config: DetectorConfig = DEFAULT_CONFIG) -> PlateCandidate | None:
    """Score every plausible contour, including ones without four vertices."""
    area = abs(float(cv2.contourArea(contour)))
    image_area = float(text_edges.size)
    if area < minimum_area or area > image_area * config.max_area_fraction:
        return None
    rectangle = cv2.minAreaRect(contour)
    (_, _), (rect_width, rect_height), _ = rectangle
    if min(rect_width, rect_height) < 8:
        return None
    tl, tr, br, bl = order_points(cv2.boxPoints(rectangle))
    aspect = float(np.linalg.norm(tr - tl) / max(np.linalg.norm(bl - tl), 1))
    if not config.min_aspect <= aspect <= config.max_aspect:
        return None
    rectangularity = min(1.0, area / (rect_width * rect_height))
    hull_area = abs(cv2.contourArea(cv2.convexHull(contour)))
    solidity = min(1.0, area / max(hull_area, 1.0))
    if rectangularity < 0.35 or solidity < 0.45:
        return None
    approximation = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
    x, y, width, height = cv2.boundingRect(contour)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [contour - np.array([[[x, y]]])], -1, 255, -1)
    # Exclude the border itself: an empty rectangle must not look like text.
    margin = max(2, round(min(rect_width, rect_height) * 0.08))
    mask = cv2.erode(mask, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0)
    interior_pixels = cv2.countNonZero(mask)
    density = cv2.countNonZero(cv2.bitwise_and(text_edges[y:y + height, x:x + width], mask)) / max(interior_pixels, 1)
    features = {"area": area, "aspect_ratio": aspect, "rectangularity": rectangularity,
                "solidity": solidity, "approximation": float(len(approximation)), "edge_density": density}
    scores = {
        "area": min(1.0, area / (minimum_area * 4)) * _band_score(area / image_area, *config.area_band),
        "aspect_ratio": _band_score(aspect, *config.aspect_band),
        "rectangularity": rectangularity,
        "solidity": solidity,
        "approximation": 1.0 / (1.0 + abs(len(approximation) - 4)),
        "edge_density": _band_score(density, 0.005, 0.05, 0.25, 0.50),
    }
    total = sum(weight * scores[key] for key, weight in zip(FEATURE_WEIGHTS, config.weights))
    return PlateCandidate(contour, (x, y, width, height), features, scores, total)


def _eligible(candidate: PlateCandidate, config: DetectorConfig = DEFAULT_CONFIG) -> bool:
    return candidate.score >= config.minimum_score and candidate.features["edge_density"] >= config.minimum_edge_density


def _deduplicate(candidates: list[PlateCandidate]) -> list[PlateCandidate]:
    unique: list[PlateCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        x, y, w, h = candidate.bbox
        duplicate = False
        for other in unique:
            ox, oy, ow, oh = other.bbox
            intersection = max(0, min(x + w, ox + ow) - max(x, ox)) * max(0, min(y + h, oy + oh) - max(y, oy))
            if intersection / max(w * h + ow * oh - intersection, 1) > 0.85:
                duplicate = True
                break
        if not duplicate:
            unique.append(candidate)
    return unique


def refine_candidate(candidate: PlateCandidate, gray: np.ndarray, text_edges: np.ndarray,
                     minimum_area: float, canny_low: int, canny_high: int,
                     config: DetectorConfig = DEFAULT_CONFIG) -> PlateCandidate | None:
    """Find the actual plate boundary inside a padded ROI; restore global coordinates."""
    x, y, width, height = candidate.bbox
    pad = max(4, round(min(width, height) * 0.25))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(gray.shape[1], x + width + pad), min(gray.shape[0], y + height + pad)
    roi = cv2.GaussianBlur(gray[y0:y1, x0:x1], (3, 3), 0)
    local_edges = cv2.Canny(roi, canny_low, canny_high)
    threshold = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    boundaries: list[PlateCandidate] = []
    for binary in (local_edges, threshold):
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for local_contour in contours:
            lx, ly, lw, lh = cv2.boundingRect(local_contour)
            # The crop border is not evidence of a physical plate boundary.
            if lx <= 0 or ly <= 0 or lx + lw >= roi.shape[1] or ly + lh >= roi.shape[0]:
                continue
            contour = local_contour + np.array([[[x0, y0]]], dtype=np.int32)
            area = abs(cv2.contourArea(contour))
            if not 0.15 * candidate.features["area"] <= area <= 1.3 * candidate.features["area"]:
                continue
            corners = _four_corners(contour, config)
            if corners is None:
                continue
            refined = score_contour(contour, text_edges, minimum_area, config)
            if refined is not None and _eligible(refined, config):
                refined.corners = corners
                refined.corner_source = "roi"
                boundaries.append(refined)
    return max(boundaries, key=lambda item: item.score, default=None)


def detect_plate(image_bgr: np.ndarray, kernel_size: int, canny_low: int,
                 canny_high: int, minimum_area: int,
                 config: DetectorConfig = DEFAULT_CONFIG) -> DetectionResult:
    """Rank candidates by all six features, then resolve corners before homography."""
    scaled_minimum = max(20.0, minimum_area * image_bgr.shape[0] * image_bgr.shape[1] / REFERENCE_IMAGE_AREA)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    kernel_size = max(1, int(kernel_size) | 1)
    low, high = sorted((canny_low, canny_high))
    blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    edges = cv2.Canny(blurred, low, high)
    # Preserve small text edges even if the user chooses strong boundary blur.
    text_edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), low, high)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    candidates = []
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 7)
    otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    for binary in (edges, closed, threshold, otsu):
        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            candidate = score_contour(contour, text_edges, scaled_minimum, config)
            if candidate is not None:
                candidates.append(candidate)
    ranked = _deduplicate(candidates)
    result = DetectionResult(candidates=ranked)
    for candidate in [item for item in ranked if _eligible(item, config)][:MAX_REFINEMENT_CANDIDATES]:
        corners = _four_corners(candidate.contour, config)
        if (corners is not None and candidate.features["rectangularity"] >= config.direct_rectangularity
                and candidate.features["solidity"] >= config.direct_solidity):
            candidate.corners = corners
            candidate.corner_source = "direct"
        else:
            refined = refine_candidate(candidate, gray, text_edges, scaled_minimum, low, high, config)
            if refined is None:
                continue
            candidate.corners = refined.corners
            candidate.corner_source = refined.corner_source
        result.selected = candidate
        break
    return result


def auto_detect_plate(image_bgr: np.ndarray, kernel_size: int, canny_low: int,
                      canny_high: int, minimum_area: int) -> np.ndarray | None:
    """Compatibility entry point for callers that only need four corners."""
    return detect_plate(image_bgr, kernel_size, canny_low, canny_high, minimum_area).corners
