import unittest

import cv2
import numpy as np

from plate_detection import (
    DATASET_CONFIG, FEATURE_WEIGHTS, detect_plate, order_points, refine_candidate,
    score_contour, validate_quadrilateral,
)


def plate_scene():
    image = np.full((400, 700, 3), 40, np.uint8)
    # A larger empty rectangle competes with the smaller, textured plate.
    cv2.rectangle(image, (25, 25), (350, 175), (235, 235, 235), -1)
    cv2.rectangle(image, (420, 245), (660, 355), (235, 235, 235), -1)
    cv2.putText(image, "AB 1234", (433, 312), cv2.FONT_HERSHEY_SIMPLEX,
                1.15, (10, 10, 10), 3, cv2.LINE_AA)
    return image


class PlateDetectionTests(unittest.TestCase):
    def test_experimental_preset_preserves_text_and_blank_behavior(self):
        result = detect_plate(plate_scene(), 3, 20, 80, 150, config=DATASET_CONFIG)
        self.assertIsNotNone(result.corners)
        centre = result.corners.mean(axis=0)
        self.assertTrue(420 < centre[0] < 660 and 245 < centre[1] < 355)
        blank = np.full((400, 700, 3), 40, np.uint8)
        cv2.rectangle(blank, (150, 140), (550, 310), (235, 235, 235), -1)
        self.assertIsNone(detect_plate(blank, 3, 20, 80, 150, config=DATASET_CONFIG).corners)

    def test_text_plate_beats_larger_blank_rectangle(self):
        result = detect_plate(plate_scene(), 5, 35, 110, 150)
        self.assertIsNotNone(result.selected)
        centre = result.corners.mean(axis=0)
        self.assertTrue(420 < centre[0] < 660 and 245 < centre[1] < 355)
        self.assertEqual(result.selected.corner_source, "direct")
        self.assertTrue(validate_quadrilateral(result.corners)[0])
        self.assertEqual(list(result.selected.scores), list(FEATURE_WEIGHTS))
        self.assertAlmostEqual(result.selected.score, sum(
            result.selected.scores[key] * weight for key, weight in FEATURE_WEIGHTS.items()))
        scores = [candidate.score for candidate in result.candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_blank_image_and_empty_rectangle_do_not_produce_a_plate(self):
        image = np.full((400, 700, 3), 40, np.uint8)
        self.assertIsNone(detect_plate(image, 5, 35, 110, 150).corners)
        cv2.rectangle(image, (150, 140), (550, 310), (235, 235, 235), -1)
        self.assertIsNone(detect_plate(image, 5, 35, 110, 150).corners)

    def test_roi_search_restores_global_coordinates(self):
        image = plate_scene()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 35, 110)
        # An irregular proposal encloses a real plate, but is not its boundary.
        proposal = np.array([[[400, 235]], [[540, 220]], [[680, 235]],
                             [[685, 365]], [[535, 380]], [[402, 365]]], np.int32)
        candidate = score_contour(proposal, edges, 20)
        self.assertIsNotNone(candidate)
        refined = refine_candidate(candidate, gray, edges, 20, 35, 110)
        self.assertIsNotNone(refined)
        self.assertEqual(refined.corner_source, "roi")
        expected = np.array([[420, 245], [660, 245], [660, 355], [420, 355]], np.float32)
        self.assertLess(np.max(np.linalg.norm(refined.corners - expected, axis=1)), 5)

    def test_perspective_plate_preserves_observed_corners(self):
        plate = np.full((110, 240, 3), 235, np.uint8)
        cv2.putText(plate, "AB 1234", (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (10, 10, 10), 3, cv2.LINE_AA)
        source = np.array([[0, 0], [239, 0], [239, 109], [0, 109]], np.float32)
        expected = np.array([[165, 125], [450, 150], [420, 275], [145, 250]], np.float32)
        image = cv2.warpPerspective(plate, cv2.getPerspectiveTransform(source, expected),
                                    (650, 400), borderValue=(40, 40, 40))
        result = detect_plate(image, 5, 35, 110, 150)
        self.assertIsNotNone(result.corners)
        self.assertLess(np.max(np.linalg.norm(result.corners - expected, axis=1)), 6)

    def test_corner_validation(self):
        self.assertFalse(validate_quadrilateral(np.zeros((4, 2)))[0])
        self.assertFalse(validate_quadrilateral(np.array([[0, 0], [100, 0], [30, 30], [0, 100]]))[0])
        corners = np.array([[100, 60], [0, 0], [0, 60], [100, 0]])
        np.testing.assert_array_equal(order_points(corners), [[0, 0], [100, 0], [100, 60], [0, 60]])


if __name__ == "__main__":
    unittest.main()
