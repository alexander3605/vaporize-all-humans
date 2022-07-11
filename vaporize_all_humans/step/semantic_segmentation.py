from dataclasses import dataclass
from typing import ClassVar, Optional

import torch
import torchvision.transforms.functional as functional
from torchtyping import TensorType
from torchvision.models.segmentation import fcn_resnet50

from vaporize_all_humans.inpainting_mask import InpaintingMask
from vaporize_all_humans.step.vaporizer_step import VaporizerStep
from vaporize_all_humans.utils import (C, F, H, W, all_tensors_have_same_shape,
                                       is_image_file, psnr, timeit)


@dataclass
class SemanticSegmentation(VaporizerStep):
    """
    Semanting segmentation on each bounding box, create segmentation mask for humans
    """

    # Class index of `person`, in COCO.
    # Reference: https://pytorch.org/vision/0.11/models.html#semantic-segmentation
    PERSON_INDEX_IN_COCO: ClassVar[int] = 15

    threshold: float = 0.5
    normalize: bool = True
    process_in_batch: bool = True

    def __post_init__(self):
        # Initialize semanting segmentation model
        self.model = fcn_resnet50(pretrained=True)
        self.model.eval()

    def _inner_computation(
        self, batch: TensorType["F", "C", "H", "W"]
    ) -> TensorType["H", "W"]:
        with torch.no_grad():
            prediction = self.model(batch)["out"]

            prediction = prediction.softmax(dim=1)
            mask = (
                prediction[:, SemanticSegmentation.PERSON_INDEX_IN_COCO, ...]
                > self.threshold
            ).float()
            return mask

    @timeit("segmentation")
    def __call__(
        self,
        image: Optional[TensorType["C", "H", "W"]] = None,
        inpainting_masks: Optional[list[InpaintingMask]] = None,
    ) -> list[InpaintingMask]:
        # Normalize image
        if self.normalize:
            _image = functional.normalize(
                image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=False
            )
        else:
            _image = image.clone()

        # Predict the segmentation mask
        bounding_boxes = [m.box for m in inpainting_masks]
        crops = [_image[:, b.ymin : b.ymax, b.xmin : b.xmax] for b in bounding_boxes]
        if self.process_in_batch and all_tensors_have_same_shape(crops):
            # Case 2: process all images in a single batch, if they have the same size
            batch = torch.stack(crops, dim=0)
            segmentation_masks = torch.unbind(self._inner_computation(batch))
        else:
            # Case 3: process images individually, if they have different sizes
            segmentation_masks = [
                self._inner_computation(c.unsqueeze(dim=0))[0] for c in crops
            ]

        # Assemble the predicted mask(s) into a list of `InpaintingMask`
        for m, sm in zip(inpainting_masks, segmentation_masks):
            m.mask = sm
        return inpainting_masks
