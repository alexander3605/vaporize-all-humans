from typing import TypeVar, Any, Callable
from functools import wraps
from torchtyping import TensorType
import torch
import numpy as np
import time


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
