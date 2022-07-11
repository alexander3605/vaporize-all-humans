from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as functional
from torchtyping import TensorType

from vaporize_all_humans.utils import C, H, W, timeit
from vaporize_all_humans.inpainting_mask import InpaintingMask
from vaporize_all_humans.step.vaporizer_step import VaporizerStep
from vaporize_all_humans.config import GAUSSIAN_BLUR_ON_MERGE


@dataclass
class Blending(VaporizerStep):

    method: str = "average"
    show_inpainted_image: bool = False
    show_inpainting_mask: bool = False

    def __post_init__(self) -> None:
        self._blending_impl = BlendingEnum.from_string(self.method)
        self._output: Optional[TensorType["C", "H", "W"]] = None

    @timeit("blending")
    def __call__(
        self,
        image: Optional[TensorType["C", "H", "W"]] = None,
        inpainting_masks: Optional[list[InpaintingMask]] = None,
    ) -> list[InpaintingMask]:
        self._output, inpainted_image, inpainting_counts = self._blending_impl(
            image, inpainting_masks
        )
        _final_image = T.ToPILImage()(self._output)
        # Visualize blending results
        if (
            self.show_inpainting_mask
            and inpainted_image is not None
            and inpainting_counts is not None
        ):
            inpainted_image = T.ToPILImage()(inpainted_image)
            inpainting_counts = T.ToPILImage()(inpainting_counts)
            inpainted_image.show()
            inpainting_counts.show()
        if self.show_inpainted_image:
            _final_image.show()
        # For compatibility with other stages
        return inpainting_masks

    @property
    def output(self) -> TensorType["C", "H", "W"]:
        if self._output is None:
            raise ValueError("output has not been computed yet")
        return self._output


class BlendingImpl:
    @abstractmethod
    def __call__(
        self, image: torch.Tensor, inpainting_masks: list[InpaintingMask]
    ) -> torch.Tensor:
        pass


class AverageBlending(BlendingImpl):
    def __call__(
        self, image: torch.Tensor, inpainting_masks: list[InpaintingMask]
    ) -> torch.Tensor:
        # Assemble final image
        inpainted_image = torch.zeros_like(image)  # Store all inpainted regions here
        inpainting_counts = torch.zeros_like(
            image
        )  # Count how many pixels have been added to each spot
        for m in inpainting_masks:
            # Store the inpainted patch into a new image
            inpainted_patch = torch.where(
                m.mask.bool(), m.inpainted_image, torch.scalar_tensor(0).float()
            )
            inpainted_image[
                :, m.box.ymin : m.box.ymax, m.box.xmin : m.box.xmax
            ] += inpainted_patch
            # Update how many times each pixel in the image has been inpainted
            inpainted_count_patch = torch.where(m.mask.bool(), 1, 0)
            inpainting_counts[
                :, m.box.ymin : m.box.ymax, m.box.xmin : m.box.xmax
            ] += inpainted_count_patch
        # Average the inpaintings of all the images.
        # Set all counts at least to 1 to avoid division-by-zero
        inpainted_image /= torch.maximum(
            inpainting_counts, torch.ones_like(inpainting_counts)
        )
        # Create mask to merge original image with inpainted spots
        mask = (inpainting_counts > 0).int()
        # Create final image
        final_image = image * (1 - mask) + inpainted_image * mask

        # Do a final merge where we blur the original mask,
        # to make blending a bit smoother
        if GAUSSIAN_BLUR_ON_MERGE:
            kernel = np.ones((5, 5), np.uint8)
            mask = torch.Tensor(
                cv2.dilate(mask[0, ...].float().numpy(), kernel, iterations=1)
            ).repeat(3, 1, 1)
            mask = functional.gaussian_blur(mask, 15)
            final_image = image * (1 - mask) + final_image * mask

        return final_image, inpainted_image, inpainting_counts


class PatchwiseAssemble(BlendingImpl):
    def __call__(
        self, image: torch.Tensor, inpainting_masks: list[InpaintingMask]
    ) -> torch.Tensor:
        final_image = image.clone()
        for m in inpainting_masks:
            # Don't merge full patch, but only masked part.
            inpainted_patch = torch.where(m.mask.bool(), m.inpainted_image, m.image)
            final_image[
                :, m.box.ymin : m.box.ymax, m.box.xmin : m.box.xmax
            ] = inpainted_patch
        return final_image, None, None


class BlendingEnum(Enum):
    AVERAGE = AverageBlending
    PATCHWISE = PatchwiseAssemble

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def from_string(string: str) -> BlendingImpl:
        """
        Create a blending strategy from a string
        """
        try:
            return BlendingEnum[string.upper()].value()
        except KeyError as e:
            raise ValueError(f"{string} is not a valid BlendingEnum") from e
