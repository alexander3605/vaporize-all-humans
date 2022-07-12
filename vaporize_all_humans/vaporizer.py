import os
from dataclasses import dataclass
from telnetlib import GA
from typing import Any, Optional, Union

import pandas as pd
import torchvision.transforms as T
from PIL import Image
from torchtyping import TensorType

from vaporize_all_humans.config import (
    BOUNDING_BOX_INCREASE_FACTOR,
    BOUNDING_BOX_MIN_SIZE,
    CONDENSE_SEGMENTATION_MASKS,
    GAUSSIAN_BLUR_ON_MERGE,
    GOLDEN_FOLDER,
    IOU_FILTER_THRESHOLD,
)
from vaporize_all_humans.step.blending import Blending
from vaporize_all_humans.step.inpainting import Inpainting
from vaporize_all_humans.step.object_detection import ObjectDetection
from vaporize_all_humans.step.semantic_segmentation import SemanticSegmentation
from vaporize_all_humans.step.vaporizer_step import (
    CleanInpaintingMasks,
    CondenseInpaintingMasks,
    MergeOverlappingInpaintingMasks,
    ScaleBoundingBoxes,
    VaporizerStep,
)
from vaporize_all_humans.utils import C, H, W, psnr


@dataclass
class Vaporizer:

    patch_size: int = BOUNDING_BOX_MIN_SIZE
    patch_increase_factor: float = BOUNDING_BOX_INCREASE_FACTOR
    iou_filter_threshold: float = IOU_FILTER_THRESHOLD
    blur_stitching: bool = GAUSSIAN_BLUR_ON_MERGE
    patches_condensation: bool = CONDENSE_SEGMENTATION_MASKS
    show_patches: bool = False
    show_mask: bool = False
    show_result: bool = False

    def __post_init__(self) -> None:

        ####################
        ### Define steps ###
        ####################

        self.steps: list[VaporizerStep] = []

        # Object detection
        self.steps += [
            ObjectDetection(),
            ScaleBoundingBoxes(
                bounding_box_increase_factor=self.patch_increase_factor,
                bounding_box_minimum_size=self.patch_size,
            ),
        ]
        # Semanting segmentation
        self.steps += [
            SemanticSegmentation(),
            CleanInpaintingMasks(),
            MergeOverlappingInpaintingMasks(),
        ]
        if self.patches_condensation:
            self.steps += [
                CondenseInpaintingMasks(),
                MergeOverlappingInpaintingMasks(),
            ]
        # Inpainting
        self.steps += [
            Inpainting(),
            Blending("average", self.show_result, self.show_mask),
        ]

        ####################
        ####################
        ####################

        assert isinstance(
            self.steps[-1], Blending
        ), f"the last step of the vaporizer pipeline must be a Blending, not {type(self.steps[-1])}"
        # Store result for fast retrieval
        self._output: Optional[dict[str, Any]] = None

    @property
    def output(self) -> TensorType["C", "H", "W"]:
        if self._output is None:
            raise ValueError("output has not been computed yet")
        return self._output

    @property
    def dataframe(self) -> TensorType["C", "H", "W"]:
        if self._output is None:
            raise ValueError("output has not been computed yet")
        rows = []
        for image, result in self._output.items():
            rows += [[image, result["psnr"], len(result["inpaintings"])]]
        return pd.DataFrame(rows, columns=["image", "psnr", "number_of_inpaintings"])

    def _load_image(self, image_path: str) -> TensorType["C", "H", "W"]:
        return T.ToTensor()(Image.open(image_path))

    def _psnr(
        self, image_path: str, predicted_image: TensorType["C", "H", "W"]
    ) -> float:
        result_psnr = -1
        if os.path.basename(image_path) in os.listdir(GOLDEN_FOLDER):
            golden = Image.open(
                os.path.join(GOLDEN_FOLDER, os.path.basename(image_path))
            )
            result_psnr = psnr(predicted_image, T.ToTensor()(golden))
            print(f"PSNR for {image_path}={result_psnr:.4f} dB")
        return result_psnr

    def __call__(self, image_paths: Union[str, list[str]]) -> None:
        # Process a list of image paths
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        self._output = {}
        for image_path in image_paths:
            print(f"\nVAPORIZE HUMANS IN {image_path}!")

            # Load the image as a tensor
            image = self._load_image(image_path)
            # Compute each step in the pipeline
            inpainting_masks = None
            for stage in self.steps:
                inpainting_masks = stage(image=image, inpainting_masks=inpainting_masks)
            # Obtain the final image from the blending step
            predicted_image = self.steps[-1].output
            # Visualize inpainted masks
            if self.show_patches:
                for m in inpainting_masks:
                    m.show()
            # Store result
            self._output[image_path] = {
                "inpaintings": inpainting_masks,
                "predicted_image": predicted_image,
                "source_image": image,
            }
            # Compute PSNR w.r.t. manually retouched image
            self._output[image_path] |= {
                "psnr": self._psnr(image_path, predicted_image)
            }

        return self._output
