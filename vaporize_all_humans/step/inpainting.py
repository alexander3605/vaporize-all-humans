import os
from dataclasses import dataclass
from typing import ClassVar, Optional
import torch
import torchvision.transforms.functional as functional
import yaml
from omegaconf import OmegaConf
from torchtyping import TensorType

from external.lama.saicinpainting.training.trainers.default import (
    DefaultInpaintingTrainingModule,
)
from vaporize_all_humans.utils import C, H, W, all_tensors_have_same_shape, timeit
from vaporize_all_humans.inpainting_mask import InpaintingMask
from vaporize_all_humans.step.vaporizer_step import VaporizerStep


@dataclass
class Inpainting(VaporizerStep):
    """
    Vaporize humans
    """

    LAMA_CONFIG_DIR: ClassVar[str] = "external/lama/big-lama"
    # LAMA_CONFIG_DIR = "external/lama/LaMa_models/lama-places/lama-fourier"
    LAMA_CHECKPOINT: ClassVar[str] = "best.ckpt"
    PAD_TO_MODULO: ClassVar[int] = 8

    process_in_batch: bool = True

    def __post_init__(self):

        train_config_path = os.path.join(Inpainting.LAMA_CONFIG_DIR, "config.yaml")
        with open(train_config_path, "r") as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
            train_config.training_model.predict_only = True
            train_config.visualizer.kind = "noop"
            checkpoint_path = os.path.join(
                self.LAMA_CONFIG_DIR, "models", Inpainting.LAMA_CHECKPOINT
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

    def _inner_computation(
        self, inpainting_masks: list[InpaintingMask]
    ) -> list[InpaintingMask]:
        images = []
        masks = []
        # Used to restore the original image shape
        original_image_size = None
        for x in inpainting_masks:
            # Process each image to create a batch
            if Inpainting.PAD_TO_MODULO > 0:
                original_image_size = x.image.shape[1:]
                image = self._pad_image_to_modulo(x.image, Inpainting.PAD_TO_MODULO)
                mask = self._pad_image_to_modulo(x.mask, self.PAD_TO_MODULO)
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
        inpainting_masks: Optional[list[InpaintingMask]] = None,
    ) -> list[InpaintingMask]:
        with torch.no_grad():
            # Use a single batch if images have the same size and a flag has been specified
            # If not, process them one by one
            if self.process_in_batch and all_tensors_have_same_shape(
                [x.image for x in inpainting_masks]
            ):
                inpainting_masks = self._inner_computation(inpainting_masks)
            else:
                # Process images independently
                inpainting_masks = [
                    self._inner_computation([x])[0] for x in inpainting_masks
                ]
            return inpainting_masks
