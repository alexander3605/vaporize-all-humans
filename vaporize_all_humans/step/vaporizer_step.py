from dataclasses import dataclass
from abc import ABC
from typing import Optional
import cv2
from vaporize_all_humans.bounding_box import BoundingBox
from torchtyping import TensorType
from vaporize_all_humans.inpainting_mask import InpaintingMask
from vaporize_all_humans.config import BOUNDING_BOX_INCREASE_FACTOR, BOUNDING_BOX_MIN_SIZE, IOU_FILTER_THRESHOLD
from vaporize_all_humans.utils import timeit, C, H, W
import torch

@dataclass
class VaporizerStep(ABC):
    def __call__(
        self,
        image: Optional[TensorType["C", "H", "W"]] = None,
        inpainting_masks: Optional[list[InpaintingMask]] = None,
    ) -> list[InpaintingMask]:
        pass


@dataclass
class ScaleBoundingBoxes(VaporizerStep):
    """
    Increase the size of each box, then make it square by enforcing a minimum edge length.
    If the shortest edge is already bigger than the specified minimum size,
      do not change its size
    """

    bounding_box_increase_factor: float = BOUNDING_BOX_INCREASE_FACTOR
    bounding_box_minimum_size: int = BOUNDING_BOX_MIN_SIZE

    @timeit("scale bounding boxes")
    def __call__(
        self,
        image: Optional[TensorType["C", "H", "W"]] = None,
        inpainting_masks: Optional[list[InpaintingMask]] = None,
    ) -> list[InpaintingMask]:
        for m in inpainting_masks:
            m.box.scale(self.bounding_box_increase_factor).to_square(
                self.bounding_box_minimum_size
            )
        return inpainting_masks


@dataclass
class CleanInpaintingMasks(VaporizerStep):
    """
    Remove segmentation masks that touch the border of the box,
        as it creates artifacts when merging.
    Then slightly contract masks to clean small noisy spots,
        and expand the masks to cover a area bigger than the precise
        human contour, as it improves inpainting
    """

    erode: int = 2
    dilate: int = 15

    @timeit("clean segmentation masks")
    def __call__(
        self,
        image: Optional[TensorType["C", "H", "W"]] = None,
        inpainting_masks: Optional[list[InpaintingMask]] = None,
    ) -> list[InpaintingMask]:
        for m in inpainting_masks:
            m.remove_mask_at_the_border()
            m.opening(erode=self.erode, dilate=self.dilate)
        # Remove masks that no longer have any segmentation in them
        return [m for m in inpainting_masks if m.mask.sum() > 0]


@dataclass
class MergeOverlappingInpaintingMasks(VaporizerStep):
    """
    Merge bounding boxes with IoU above a threshold.
    Keep the mask with biggest segmentation mask
    """

    @timeit("merge overlapping segmentation masks")
    def __call__(
        self,
        image: Optional[TensorType["C", "H", "W"]] = None,
        inpainting_masks: Optional[list[InpaintingMask]] = None,
    ) -> list[InpaintingMask]:
        filtered_masks = []
        for i, m1 in enumerate(inpainting_masks):
            keep_m1 = True
            for m2 in inpainting_masks[i + 1 :]:
                # If the two boxes overlap more than a threshold,
                # keep the box whose mask has bigger area.

                # FIXME: if `iou(m1, m2) > IOU_FILTER_THRESHOLD` and `iou(m2, m3) > IOU_FILTER_THRESHOLD`,
                # but `iou(m1, m3) < IOU_FILTER_THRESHOLD`, we currently keep just m3, but we should keep also m1 or m2
                if (
                    m1.box.iou(m2.box) > IOU_FILTER_THRESHOLD
                    and m1.mask.sum().numpy() < m2.mask.sum().numpy()
                ):
                    keep_m1 = False
            if keep_m1:
                filtered_masks += [m1]
        print(f"removed {len(inpainting_masks) - len(filtered_masks)} masks")
        return filtered_masks


@dataclass
class CondenseInpaintingMasks(VaporizerStep):
    """
    Merge all segmentation masks into one, then create bounding boxes over white regions.
    Then do inpainting on these new bounding boxes.
    This procedure creates better inpainting masks, by merging masks of humans close to each other
    """

    @timeit("condense segmentation masks")
    def __call__(
        self,
        image: Optional[TensorType["C", "H", "W"]] = None,
        inpainting_masks: Optional[list[InpaintingMask]] = None,
    ) -> list[InpaintingMask]:
        # First, merge all segmentation masks into a unique mask
        total_mask = torch.zeros(image.shape).bool()[0, ...]
        for m in inpainting_masks:
            # Don't merge the full patch, but only the masked parts
            total_mask[
                m.box.ymin : m.box.ymax, m.box.xmin : m.box.xmax
            ] |= m.mask.bool()
        total_mask = total_mask.float()
        total_mask_byte = (total_mask * 255).byte().numpy()

        # Compute bounding boxes over this mask
        contours = cv2.findContours(
            total_mask_byte, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = contours[0] if len(contours) == 2 else contours[1]
        boxes = []
        for i, c in enumerate(contours):
            x, y, w, h = cv2.boundingRect(c)
            boxes += [
                BoundingBox(x, x + w, y, y + h, i, 1, list(total_mask_byte.shape))
            ]
        # Increase size of bounding boxes
        for b in boxes:
            b.scale(BOUNDING_BOX_INCREASE_FACTOR).to_square(BOUNDING_BOX_MIN_SIZE)

        # Create new segmentation masks
        new_masks = []
        for b in boxes:
            image_patch = image[:, b.ymin : b.ymax, b.xmin : b.xmax]
            mask = total_mask[b.ymin : b.ymax, b.xmin : b.xmax]
            new_masks += [InpaintingMask(b, image_patch, mask, None)]
        return new_masks
