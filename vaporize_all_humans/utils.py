import os
import time
from argparse import Namespace
from functools import wraps
from typing import Any, Callable, TypeVar, Union

import numpy as np
import torch
from torchtyping import TensorType

from vaporize_all_humans.config import DATA_FOLDER
from vaporize_all_humans.utils import is_image_file

F = TypeVar("F")  # Frames
C = TypeVar("C")  # Channels
W = TypeVar("W")  # Width
H = TypeVar("H")  # Height


IMG_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".tif",
]
IMG_EXTENSIONS += [x.upper() for x in IMG_EXTENSIONS]


def is_image_file(filename: str) -> bool:
    """
    Checks whether a filename has an image extension, i.e. one of:
        ".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG", ".ppm", ".PPM", ".bmp", ".BMP",
        ".tif", ".TIF",

    :param filename: The file name of the image
    :return: Whether the filename has any one of the supported file extensions
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def timeit(name: str = ""):
    def _timeit(f: Callable):
        @wraps(f)
        def wrap(*args: Any, **kwargs: Any):
            s = time.time()
            result = f(*args, **kwargs)
            e = time.time()
            function_name = name if name else f.__name__
            print(f"function: {function_name}, time: {e - s:.2f} sec")
            return result

        return wrap

    return _timeit


def psnr(
    x: TensorType["C", "H", "W", float],
    y: TensorType["C", "H", "W", float],
    peak: float = 255,
):
    return 10 * np.log10(peak**2 / torch.mean((x - y) ** 2).numpy())


def all_tensors_have_same_shape(images: list[torch.Tensor]) -> bool:
    first_size = images[0].shape
    return all([x.shape == first_size for x in images])


def get_demo_images() -> list[str]:
    return sorted(
        [
            os.path.join(DATA_FOLDER, x)
            for x in os.listdir(DATA_FOLDER)
            if is_image_file(x)
        ]
    )


def validate_input_images(image_paths: Union[str, list[str]]) -> None:
    for img in image_paths:
        assert os.path.exists(img), f"input image {img} does not exists"
        assert is_image_file(
            img
        ), f"input file {img} is not a valid image, supported formats are {IMG_EXTENSIONS}"


def get_input_images(args: Namespace) -> list[str]:
    image_paths = get_demo_images() if args.demo else args.input
    image_paths(image_paths)
    return image_paths
