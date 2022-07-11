#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 08:09:11 2022

@author: albertoparravicini
"""

import torch
import os
from PIL import Image
from typing import Union, Optional
import torchvision.transforms as T
import torchvision.transforms.functional as functional
import numpy as np
from torchtyping import TensorType
from omegaconf import OmegaConf
import yaml
import cv2
from torchvision.utils import make_grid
from copy import copy
from abc import abstractmethod, ABC
from enum import Enum

from dataclasses import dataclass
from torchvision.models.segmentation import fcn_resnet50
from vaporize_all_humans.utils import is_image_file, timeit, all_tensors_have_same_shape, psnr, F, C, H, W
from external.lama.saicinpainting.training.trainers.default import (
    DefaultInpaintingTrainingModule,
)

DATA_FOLDER = "data/samples"
GOLDEN_FOLDER = os.path.join(DATA_FOLDER, "golden")
BOUNDING_BOX_INCREASE_FACTOR = 0.5
BOUNDING_BOX_MIN_SIZE = 224
IOU_FILTER_THRESHOLD = 0.5
GAUSSIAN_BLUR_ON_MERGE = False
CONDENSE_SEGMENTATION_MASKS = True

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


@dataclass
class InpainingMask:
    box: BoundingBox
    image: TensorType["C", "H", "W", float]
    mask: TensorType["H", "W", float] = None
    inpainted_image: TensorType["C", "H", "W", float] = None

    def __post_init__(self) -> None:
        assert self.box.shape == self.image.shape[-2:], f"{self.box.shape=} != {self.image.shape[-2:]=}"
        if self.mask is not None:
            assert self.mask.shape == self.image.shape[-2:], f"{self.mask.shape=} != {self.image.shape[-2:]=}"
        if self.inpainted_image is not None:
            assert self.inpainted_image.shape == self.image.shape, f"{self.inpainted_image.shape=} != {self.image.shape=}"

    def remove_mask_at_the_border(self) -> "SemanticSegmentation":
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

    def opening(self, erode: int = 2, dilate: int = 15) -> "InpainingMask":
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

@dataclass
class VaporizerStage(ABC):
    def __call__(
        self,
        image: Optional[TensorType["C", "H", "W"]] = None,
        inpainting_masks: Optional[list[InpainingMask]] = None
    ) -> list[InpainingMask]:
        pass


@dataclass
class ObjectDetection(VaporizerStage):

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

    def _initialize_segmentation_masks(self, image: TensorType["C", "H", "W"], boxes: list[BoundingBox]) -> list[InpainingMask]:
        return [InpainingMask(b, image[:, b.ymin : b.ymax, b.xmin : b.xmax], None, None) for b in boxes]

    @timeit("object detection")
    def __call__(
        self,
        image: Optional[TensorType["C", "H", "W"]] = None,
        inpainting_masks: Optional[list[InpainingMask]] = None
    ) -> list[InpainingMask]:
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


@dataclass
class SemanticSegmentation(VaporizerStage):

    # Class index of `person`, in COCO.
    # Reference: https://pytorch.org/vision/0.11/models.html#semantic-segmentation
    PERSON_INDEX_IN_COCO = 15

    threshold: float = 0.5
    normalize: bool = True
    process_in_batch: bool = True

    def __post_init__(self):
        # Initialize semanting segmentation model
        self.model = fcn_resnet50(pretrained=True)
        self.model.eval()

    def _inner_computation(
        self,
        batch: TensorType["F", "C", "H", "W"]
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
        inpainting_masks: Optional[list[InpainingMask]] = None
    ) -> list[InpainingMask]:
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
            segmentation_masks = [self._inner_computation(c.unsqueeze(dim=0))[0] for c in crops]

        # Assemble the predicted mask(s) into a list of `InpainingMask`
        for m, sm in zip(inpainting_masks, segmentation_masks):
            m.mask = sm
        return inpainting_masks


@dataclass
class Inpainting(VaporizerStage):

    LAMA_CONFIG_DIR = "external/lama/big-lama"
    # LAMA_CONFIG_DIR = "external/lama/LaMa_models/lama-places/lama-fourier"
    LAMA_CHECKPOINT = "best.ckpt"
    PAD_TO_MODULO = 8

    process_in_batch: bool = True

    def __post_init__(self):

        train_config_path = os.path.join(Inpainting.LAMA_CONFIG_DIR, "config.yaml")
        with open(train_config_path, "r") as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
            train_config.training_model.predict_only = True
            train_config.visualizer.kind = "noop"
            checkpoint_path = os.path.join(
                Inpainting.LAMA_CONFIG_DIR, "models", Inpainting.LAMA_CHECKPOINT
            )
            self.model = self._load_checkpoint(train_config, checkpoint_path)
        self.model.freeze()

    def _load_checkpoint(
        self, train_config, checkpoint_path, map_location="cpu", strict=False
    ):
        kwargs = dict(train_config.training_model)
        kwargs["use_ddp"] = False
        kwargs.pop("kind")
        model: torch.nn.Module = DefaultInpaintingTrainingModule(train_config, **kwargs)
        state = torch.load(checkpoint_path, map_location=map_location)
        model.load_state_dict(state["state_dict"], strict=strict)
        model.on_load_checkpoint(state)
        return model
    
    def _ceil_modulo(self, x, mod):
        if x % mod == 0:
            return x
        return (x // mod + 1) * mod

    def _pad_image_to_modulo(self, image: torch.Tensor, modulo: int):
        height, width = image.shape[-2:]
        out_height = self._ceil_modulo(height, modulo)
        out_width = self._ceil_modulo(width, modulo)
        padded_shape = [0, 0, out_width - width, out_height - height]
        take_first_dim = False
        if len(image.shape) == 2:
            image = image.unsqueeze(dim=0)
            take_first_dim = True
        image = functional.pad(image, padded_shape, padding_mode="symmetric")
        if take_first_dim:
            image = image[0]
        return image

    def _inner_computation(self, inpainting_masks: list[InpainingMask]) -> list[InpainingMask]:
        images = []
        masks = []
        # Used to restore the original image shape
        original_image_size = None
        for x in inpainting_masks:
            # Process each image to create a batch
            if Inpainting.PAD_TO_MODULO > 0:
                original_image_size = x.image.shape[1:]
                image = self._pad_image_to_modulo(x.image, Inpainting.PAD_TO_MODULO)
                mask = self._pad_image_to_modulo(x.mask, Inpainting.PAD_TO_MODULO)
            else:
                image = x.image
                mask = x.mask
            images += [image]
            masks += [mask.unsqueeze(dim=0)]
        # Assemble the batch
        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)
        batch = {"image": images, "mask": masks}
        # Compute inpainting
        batch = self.model(batch)
        # Retrieve the result, and ensure it has the correct size
        inpainted = batch["inpainted"]
        if original_image_size is not None:
            original_height, original_width = original_image_size
            inpainted = inpainted[..., :original_height, :original_width]
        for i, x in enumerate(inpainting_masks):
            x.inpainted_image = inpainted[i, ...]
        return inpainting_masks

    @timeit("inpainting")
    def __call__(
        self,
        image: Optional[TensorType["C", "H", "W"]] = None,
        inpainting_masks: Optional[list[InpainingMask]] = None
    ) -> list[InpainingMask]:
        with torch.no_grad():
            # Use a single batch if images have the same size and a flag has been specified
            # If not, process them one by one
            if self.process_in_batch and all_tensors_have_same_shape(
                [x.image for x in inpainting_masks]
            ):
                inpainting_masks = self._inner_computation(inpainting_masks)
            else:
                # Process images independently
                inpainting_masks = [self._inner_computation([x])[0] for x in inpainting_masks]
            return inpainting_masks


@dataclass
class Blending(VaporizerStage):

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
        inpainting_masks: Optional[list[InpainingMask]] = None
    ) -> list[InpainingMask]:       
        self._output, inpainted_image, inpainting_counts = self._blending_impl(image, inpainting_masks)
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
        self, image: torch.Tensor, inpaintings: list[InpainingMask]
    ) -> torch.Tensor:
        pass


class AverageBlending(BlendingImpl):
    def __call__(
        self, image: torch.Tensor, inpaintings: list[InpainingMask]
    ) -> torch.Tensor:
        # Assemble final image
        inpainted_image = torch.zeros_like(image)  # Store all inpainted regions here
        inpainting_counts = torch.zeros_like(
            image
        )  # Count how many pixels have been added to each spot
        for m in inpaintings:
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
        self, image: torch.Tensor, inpaintings: list[InpainingMask]
    ) -> torch.Tensor:
        final_image = image.clone()
        for m in inpaintings:
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
            
    
class Vaporizer:

    def __init__(self) -> None:
        
        # self.stages = [
        #     # Initialize model for object detection
        #     ObjectDetection(),
        #     # Initialize model for semanting segmentation
        #     SemanticSegmentation(),
        #     # Initialize model for inpaining
        #     Inpainting(),
        #     # How to blend inpainted patches to assemble the final image
        #     Blending("average", True, True)
        # ]
        
        # Initialize model for object detection
        self.object_detection = ObjectDetection()
        # Initialize model for semanting segmentation
        self.semanting_segmentation = SemanticSegmentation()
        # Initialize model for inpaining
        self.inpainting = Inpainting()
        # How to blend inpainted patches to assemble the final image
        self.blending = Blending("average", True, True)
        
    @timeit()
    def _merge_similar_masks(self, masks: list[InpainingMask]) -> list[InpainingMask]:
        filtered_masks = []
        for i, m1 in enumerate(masks):
            keep_m1 = True
            for m2 in masks[i + 1 :]:
                # If the two boxes overlap more than `IOU_FILTER_THRESHOLD`,
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
        print(f"removed {len(masks) - len(filtered_masks)} masks")
        return filtered_masks
        
    def __call__(self, image_paths: Union[str, list[str]]) -> None:
        # Process a list of image paths
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        result = {}
        for image_path in image_paths:

            image = self.load_image(image_path)

            # Get bounding boxes of humans
            inpainting_masks = self.object_detection(image=image)

            # Increase each box by 50%, then make it square (and at least 224x224 in size)
            for m in inpainting_masks:
                m.box.scale(BOUNDING_BOX_INCREASE_FACTOR).to_square(BOUNDING_BOX_MIN_SIZE)

            # Semanting segmentation on each bounding box, create segmentation mask for humans
            inpainting_masks = self.semanting_segmentation(image=image, inpainting_masks=inpainting_masks)

            # Clean and expand segmentation masks, to improve inpainting
            # Also remove any segmentation mask that touches the border
            inpainting_masks = self._clean_masks(inpainting_masks)

            # TODO: handle shadows: identify each contiguous region in the mask, flip it vertically, shift it down, and extrude
            # To identify regions, use the original bounding boxes. Extend them by 10% to be safe, but only left/right/top, not bottom
            # TODO: rotate all boxes around bottom, starting from 60-60 angle. Look for min average brightness, that's probably the shadow angle

            # Merge blocks with IoU above `IOU_FILTER_THRESHOLD`.
            # Keep the mask with biggest segmentation mask
            inpainting_masks = self._merge_similar_masks(inpainting_masks)

            # Merge all segmentation masks into one, then create bounding boxes over white regions
            # Then do inpainting on these new bounding boxes.
            # Better cause we don't have to worry about merging original boxes
            if CONDENSE_SEGMENTATION_MASKS:
                inpainting_masks = self._create_condensed_masks(image, inpainting_masks)

            # Vaporize humans
            inpainting_masks = self.inpainting(inpainting_masks=inpainting_masks)

            # Visualize inpainted masks
            # for m in segmentation_masks:
            #     m.show()

            # Assemble final image
            inpainting_masks = self.blending(image, inpainting_masks)
            final_image = self.blending.output
            
            # Store result
            result[image_path] = {"inpaintings": inpainting_masks, "final_image": final_image, "original_image": image}

            # Compute PSNR w.r.t. manually retouched image
            if os.path.basename(image_path) in os.listdir(GOLDEN_FOLDER):
                golden = T.ToTensor()(
                    Image.open(os.path.join(GOLDEN_FOLDER, os.path.basename(image_path)))
                )
                _psnr = psnr(final_image, golden)
                result[image_path] |= {"psnr": _psnr}
                print(f"PSNR={_psnr:.4f} dB")
            
        return result

    def load_image(self, image_path: str) -> TensorType["C", "H", "W"]:
        return T.ToTensor()(Image.open(image_path))
    
    @timeit()
    def _clean_masks(self, masks: list[InpainingMask], erode: int = 2, dilate: int = 15) -> list[InpainingMask]:
        """
        Remove segmentation masks that touch the border of the box,
          as it creates artifacts when merging.
        Then slightly contract masks to clean small noisy spots,
          and expand the masks to cover a area bigger than the precise
          human contour, as it improves inpainting
        """
        for m in masks:
            m.remove_mask_at_the_border()
            m.opening(erode=erode, dilate=dilate)
        # Remove masks that no longer have any segmentation in them
        return [m for m in masks if m.mask.sum() > 0]

    @timeit()
    def _create_condensed_masks(self, image: torch.Tensor, masks: list[InpainingMask]) -> list[InpainingMask]:
        """
        Merge all segmentation masks into one, 
          then create bounding boxes over white regions.
        Then do inpainting on these new bounding boxes. 
        This procedure creates better inpainting masks, 
          by merging masks of humans close to each other
        """
       
        # First, merge all segmentation masks into a unique mask
        total_mask = torch.zeros(image.shape).bool()[0, ...]
        for m in masks:
            # Don't merge the full patch, but only the masked parts
            total_mask[
                m.box.ymin : m.box.ymax, m.box.xmin : m.box.xmax
            ] |= m.mask.bool()
        total_mask = total_mask.float()
        total_mask_byte = (total_mask * 255).byte().numpy()
            
        # Compute bounding boxes over this mask
        contours = cv2.findContours(total_mask_byte, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        boxes = []
        for i, c in enumerate(contours):
            x, y, w, h = cv2.boundingRect(c)
            boxes += [BoundingBox(x, x + w, y, y + h, i, 1, list(total_mask_byte.shape))]
        # Increase size of bounding boxes
        for b in boxes:
            b.scale(BOUNDING_BOX_INCREASE_FACTOR).to_square(BOUNDING_BOX_MIN_SIZE)
            
        # Create new segmentation masks
        new_masks = []
        for b in boxes:
            image_patch = image[:, b.ymin : b.ymax, b.xmin : b.xmax]
            mask = total_mask[b.ymin : b.ymax, b.xmin : b.xmax]
            new_masks += [InpainingMask(b, image_patch, mask, None)]
            
        # Do a final filtering step to remove unnecessary masks
        new_masks = self._merge_similar_masks(new_masks)
        return new_masks
    
   
#%% Vaporize humans
if __name__ == "__main__":

    # Images
    images = sorted(
        [
            os.path.join(DATA_FOLDER, x)
            for x in os.listdir(DATA_FOLDER)
            if is_image_file(x)
        ]
    )[2:3]
    
    # Inference
    vaporizer = Vaporizer()
    result = vaporizer(images)

       