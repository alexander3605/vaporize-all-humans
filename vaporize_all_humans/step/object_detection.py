from dataclasses import dataclass
from vaporize_all_humans.step.vaporizer_step import VaporizerStep
from vaporize_all_humans.bounding_box import BoundingBox
from vaporize_all_humans.inpainting_mask import InpaintingMask
from torchtyping import TensorType
from typing import Optional
import torch
import numpy as np
import torchvision.transforms as T
from vaporize_all_humans.utils import C, H, W, timeit


@dataclass
class ObjectDetection(VaporizerStep):
    """
    Get bounding boxes of humans
    """

    CHECKPOINT_PATH = "checkpoints/yolov5s.pt"
    threshold: float = 0.5

    def __post_init__(self) -> None:
        # Download the model (this will be cached for future reuse)
        self.model = torch.hub.load(
            "ultralytics/yolov5", "yolov5s", pretrained=True, verbose=False
        )
        # Only detect persons
        self.model.classes = [0]
        # Somewhat low confidence, to detect small persons
        self.model.conf = self.threshold
        # Smaller IoU threhsold
        self.model.iou = 0.7

    def _initialize_segmentation_masks(
        self, image: TensorType["C", "H", "W"], boxes: list[BoundingBox]
    ) -> list[InpaintingMask]:
        return [
            InpaintingMask(b, image[:, b.ymin : b.ymax, b.xmin : b.xmax], None, None)
            for b in boxes
        ]

    @timeit("object detection")
    def __call__(
        self,
        image: Optional[TensorType["C", "H", "W"]] = None,
        inpainting_masks: Optional[list[InpaintingMask]] = None,
    ) -> list[InpaintingMask]:
        # Predict bounding boxes of people.
        # We do object detection at full resolution to detect all those small pesky humans
        with torch.no_grad():
            # This model requires a HWC numpy matrix to return a dataframe,
            # otherwise it returns a tensor
            results = self.model(np.array(T.ToPILImage()(image)), size=image.shape[-1])
        # Get bounding boxes
        boxes_dataframe = results.pandas().xyxy[0]
        boxes = [
            BoundingBox(
                r["xmin"],
                r["xmax"],
                r["ymin"],
                r["ymax"],
                i,
                r["confidence"],
                image.shape[-2:],
            )
            for i, r in boxes_dataframe.iterrows()
        ]
        # Initialize segmentation masks
        return self._initialize_segmentation_masks(image, boxes)
