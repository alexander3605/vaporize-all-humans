from copy import copy
from dataclasses import dataclass

import numpy as np


@dataclass
class BoundingBox:
    xmin: int
    xmax: int
    ymin: int
    ymax: int
    index: int
    confidence: float
    _original_shape: tuple[int, int]

    def __post_init__(self) -> None:
        self.xmin = int(self.xmin)
        self.xmax = int(self.xmax)
        self.ymin = int(self.ymin)
        self.ymax = int(self.ymax)
        self._original_box = copy(self)
        # Ensure boxes don't go outside the original image
        self._fix_size()

    def _fix_size(self) -> "BoundingBox":
        # Ensure boxes don't go outside the original image
        self.xmin = max(0, self.xmin)
        self.xmax = min(self.xmax, self._original_shape[1])
        self.ymin = max(0, self.ymin)
        self.ymax = min(self.ymax, self._original_shape[0])
        assert self.xmin >= 0
        assert self.xmax >= 0
        assert self.ymin >= 0
        assert self.ymax >= 0
        assert self.xmax > self.xmin
        assert self.ymax > self.ymin

    def iou(self, other: "BoundingBox") -> float:
        # Determine the (x, y)-coordinates of the intersection rectangle
        xmin_intersection = max(self.xmin, other.xmin)
        ymin_intersection = max(self.ymin, other.ymin)
        xmax_intersection = min(self.xmax, other.xmax)
        ymax_intersection = min(self.ymax, other.ymax)
        # Compute the area of intersection rectangle. The sides
        # of the intersection rectangle must both be positive
        intersection_area = max(0, xmax_intersection - xmin_intersection) * max(
            0, ymax_intersection - ymin_intersection
        )
        # Compute the area of the two bounding boxes. The union area
        # is the sum of the two areas minus the intersection area
        self_area = np.prod(self.shape)
        other_area = np.prod(other.shape)
        union_area = self_area + other_area - intersection_area
        # Compute IoU
        return intersection_area / union_area if union_area > 0 else 0

    def scale(self, scaling_factor: float) -> "BoundingBox":
        """
        Increase the size of bounding boxes by a proportional scaling factor.
        The size is increased keeping the center constant
        """
        height, width = self.shape
        xfactor = int(width * (scaling_factor / 2))
        yfactor = int(height * (scaling_factor / 2))
        self.xmin -= xfactor
        self.xmax += xfactor
        self.ymin -= yfactor
        self.ymax += yfactor
        # Ensure boxes don't go outside the original image
        self._fix_size()
        return self

    def to_square(self, minimum_length: int) -> "BoundingBox":
        height, width = self.shape
        new_height = new_width = max(height, width)
        # If we enforce the minimum length, we do a final cropping in the end
        enforce_size = False
        if new_height < minimum_length:
            new_height = new_width = minimum_length
            enforce_size = True
        delta_y = int(np.ceil((new_height - height) / 2))
        delta_x = int(np.ceil((new_width - width) / 2))
        assert delta_y >= 0
        assert delta_x >= 0
        self.xmin -= delta_x
        self.xmax += delta_x
        self.ymin -= delta_y
        self.ymax += delta_y
        # Ensure we have a perfect square, there might be issues with rounding
        if self.shape[0] != self.shape[1]:
            self._to_square_fix()
        # Do a final cropping, in case a minimum size was requested
        if enforce_size:
            delta = self.shape[0] - minimum_length
            self.xmax -= delta
            self.ymax -= delta
        assert (
            self.shape[0] == self.shape[1]
        ), f"shape should be square, not {self.shape}"
        # Ensure boxes don't go outside the original image
        # FIXME: maybe shift the box and maintain it square
        self._fix_size()
        return self

    def _to_square_fix(self):
        height, width = self.shape
        new_height, new_width = [max(height, width)] * 2
        delta_y = new_height - height
        delta_x = new_width - width
        assert delta_y >= 0
        assert delta_x >= 0
        self.xmax += delta_x
        self.ymax += delta_y

    @property
    def shape(self) -> tuple[int, int]:
        # Max coordinates are exclusive, min coordinates are inclusive
        return (self.ymax - self.ymin, self.xmax - self.xmin)

    def __str__(self) -> str:
        return f"[x=({int(self.xmin)}, {int(self.xmax)}), y=({int(self.ymin)}, {int(self.ymax)})]"

    def __repr__(self) -> str:
        return str(self)
