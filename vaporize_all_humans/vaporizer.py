import os
from typing import Any, Optional, Union

import pandas as pd
import torchvision.transforms as T
from PIL import Image
from torchtyping import TensorType

from vaporize_all_humans.config import (CONDENSE_SEGMENTATION_MASKS,
                                        GOLDEN_FOLDER)
from vaporize_all_humans.step.blending import Blending
from vaporize_all_humans.step.inpainting import Inpainting
from vaporize_all_humans.step.object_detection import ObjectDetection
from vaporize_all_humans.step.semantic_segmentation import SemanticSegmentation
from vaporize_all_humans.step.vaporizer_step import (
    CleanInpaintingMasks, CondenseInpaintingMasks,
    MergeOverlappingInpaintingMasks, ScaleBoundingBoxes, VaporizerStep)
from vaporize_all_humans.utils import C, H, W, psnr


class Vaporizer:
    def __init__(self) -> None:
        # Object detection
        self.steps: list[VaporizerStep] = [ObjectDetection(), ScaleBoundingBoxes()]
        # Semanting segmentation
        self.steps += [
            SemanticSegmentation(),
            CleanInpaintingMasks(),
            MergeOverlappingInpaintingMasks(),
        ]
        if CONDENSE_SEGMENTATION_MASKS:
            self.steps += [
                CondenseInpaintingMasks(),
                MergeOverlappingInpaintingMasks(),
            ]
        # Inpainting
        self.steps += [
            Inpainting(),
            Blending("average", True, True),
        ]
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
            rows += [image, result["psnr"], len(result["inpaintings"])]
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
            # Load the image as a tensor
            image = self._load_image(image_path)
            # Compute each step in the pipeline
            inpainting_masks = None
            for stage in self.steps:
                inpainting_masks = stage(image=image, inpainting_masks=inpainting_masks)
            # Obtain the final image from the blending step
            predicted_image = self.steps[-1].output
            # Visualize inpainted masks
            # for m in inpainting_masks:
            #     m.show()
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
