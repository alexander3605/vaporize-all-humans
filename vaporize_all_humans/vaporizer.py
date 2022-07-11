import os
from typing import Union

import torchvision.transforms as T
from PIL import Image
from torchtyping import TensorType
from vaporize_all_humans.utils import psnr
from vaporize_all_humans.step.semantic_segmentation import SemanticSegmentation
from vaporize_all_humans.step.object_detection import ObjectDetection
from vaporize_all_humans.step.vaporizer_step import ScaleBoundingBoxes, CleanInpaintingMasks, MergeOverlappingInpaintingMasks, CondenseInpaintingMasks
from vaporize_all_humans.step.inpainting import Inpainting
from vaporize_all_humans.step.blending import Blending
from vaporize_all_humans.config import GOLDEN_FOLDER, CONDENSE_SEGMENTATION_MASKS


class Vaporizer:
    def __init__(self) -> None:
        # Initialize model for object detection
        self.stages = [ObjectDetection(), ScaleBoundingBoxes()]
        # Initialize model for semanting segmentation
        self.stages += [
            SemanticSegmentation(),
            CleanInpaintingMasks(),
            MergeOverlappingInpaintingMasks(),
        ]
        if CONDENSE_SEGMENTATION_MASKS:
            self.stages += [
                CondenseInpaintingMasks(),
                MergeOverlappingInpaintingMasks(),
            ]
        # Initialize model for inpaining
        self.stages += [
            Inpainting(),
            # How to blend inpainted patches to assemble the final image
            Blending("average", True, True),
        ]

        assert isinstance(
            self.stages[-1], Blending
        ), f"the last stage of the vaporizer pipeline must be a Blending, not {type(self.stages[-1])}"

    def _load_image(self, image_path: str) -> TensorType["C", "H", "W"]:
        return T.ToTensor()(Image.open(image_path))

    def _psnr(self, image_path: str, predicted_image: TensorType["C", "H", "W"]) -> float:
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
        result = {}
        for image_path in image_paths:
            # Load the image as a tensor
            image = self._load_image(image_path)
            # Compute each stage in the pipeline
            inpainting_masks = None
            for stage in self.stages:
                inpainting_masks = stage(image=image, inpainting_masks=inpainting_masks)
            # Obtain the final image from the blending stage
            predicted_image = self.stages[-1].output
            # Visualize inpainted masks
            # for m in inpainting_masks:
            #     m.show()
            # Store result
            result[image_path] = {
                "inpaintings": inpainting_masks,
                "predicted_image": predicted_image,
                "source_image": image,
            }
            # Compute PSNR w.r.t. manually retouched image
            result[image_path] |= {"psnr": self._psnr(image_path, predicted_image)}

        return result
