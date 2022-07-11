import dataclasses as dataclass

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchtyping import TensorType
from torchvision.utils import make_grid

from vaporize_all_humans.bounding_box import BoundingBox
from vaporize_all_humans.utils import C, H, W
from dataclasses import dataclass

@dataclass
class InpaintingMask:
    box: BoundingBox
    image: TensorType["C", "H", "W", float]
    mask: TensorType["H", "W", float] = None
    inpainted_image: TensorType["C", "H", "W", float] = None

    def __post_init__(self) -> None:
        assert (
            self.box.shape == self.image.shape[-2:]
        ), f"{self.box.shape=} != {self.image.shape[-2:]=}"
        if self.mask is not None:
            assert (
                self.mask.shape == self.image.shape[-2:]
            ), f"{self.mask.shape=} != {self.image.shape[-2:]=}"
        if self.inpainted_image is not None:
            assert (
                self.inpainted_image.shape == self.image.shape
            ), f"{self.inpainted_image.shape=} != {self.image.shape=}"

    def remove_mask_at_the_border(self) -> "InpaintingMask":
        assert self.mask is not None, "mask cannot be None"
        # Add 1 pixel of white border around the mask
        padding = cv2.copyMakeBorder(
            self.mask.numpy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1
        )
        h, w = padding.shape

        # Create an empty mask, we do flood fill the entire image.
        # It has to be 2 pixel largers as required by OpenCV
        mask = np.zeros([h + 2, w + 2], np.uint8)

        # Flood fill the outer white border with black.
        # Fill all pixels at the border with brightness higher than 0.5 (i.e. all whites)
        mask_filled = cv2.floodFill(padding, mask, (0, 0), 0, (0.5), (0), flags=8)[1]

        # Remove the extra border
        self.mask = torch.Tensor(mask_filled[1 : h - 1, 1 : w - 1])
        return self

    def opening(self, erode: int = 2, dilate: int = 15) -> "InpaintingMask":
        assert self.mask is not None, "mask cannot be None"
        mask = self.mask.numpy()
        if erode > 0:
            kernel = np.ones((erode, erode), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
        if dilate > 0:
            kernel = np.ones((dilate, dilate), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        self.mask = torch.Tensor(mask)
        return self

    def show(self) -> None:
        assert self.mask is not None, "mask cannot be None"
        # Overlay the mask over the original picture
        overlay = Image.new("RGB", list(self.image.shape[1:][::-1]), "red")
        image_with_overlay = T.ToPILImage()(self.image.clone())
        mask = T.ToPILImage()(self.mask * 0.3)
        image_with_overlay.paste(overlay, (0, 0), mask)
        # image_with_overlay = T.ToTensor()(Image.composite(original_image, overlay, mask).convert("RGB"))
        image_with_overlay = T.ToTensor()(image_with_overlay.convert("RGB"))
        # Stack all images into a grid
        images_stacked = [
            self.image,
            self.mask.unsqueeze(0).repeat(3, 1, 1),
            image_with_overlay,
        ]
        # Show inpainting result if available
        if self.inpainted_image is not None:
            images_stacked += [self.inpainted_image]
        grid = make_grid(images_stacked, nrow=len(images_stacked), padding=0)
        T.ToPILImage()(grid).show()
