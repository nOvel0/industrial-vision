"""Сегментация нового пятна относительно чистого кадра."""

import cv2
import numpy as np

from src.zones.geometry import polygon_to_pixels


class SpillDetector:
    def __init__(self, polygon, difference_threshold, min_area_ratio):
        self.polygon = polygon
        self.threshold = difference_threshold
        self.min_area_ratio = min_area_ratio
        self.background = None

    def reset(self):
        self.background = None

    def predict(self, frame):
        height, width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)

        mask = np.zeros_like(gray)

        if self.background is None:
            self.background = gray.copy()
            return mask

        roi = np.zeros_like(gray)

        polygon = polygon_to_pixels(
            self.polygon, width, height
        )

        cv2.fillPoly(
            roi,
            [np.array(polygon, dtype=np.int32)],
            255,
        )

        difference = cv2.absdiff(gray, self.background)

        _, changed = cv2.threshold(
            difference, self.threshold, 255, cv2.THRESH_BINARY
        )

        changed = cv2.bitwise_and(changed, roi)

        kernel = np.ones((5, 5), dtype=np.uint8)
        changed = cv2.morphologyEx(
            changed, cv2.MORPH_OPEN, kernel
        )
        changed = cv2.morphologyEx(
            changed, cv2.MORPH_CLOSE, kernel
        )

        contours, _ = cv2.findContours(
            changed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest = max(contours, key=cv2.contourArea)
            min_area = cv2.countNonZero(roi) * self.min_area_ratio

            if cv2.contourArea(largest) > min_area:
                cv2.drawContours(mask, [largest], -1, 255, -1)

        return mask