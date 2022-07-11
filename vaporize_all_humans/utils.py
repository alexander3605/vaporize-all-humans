import os
import time
from argparse import Namespace
from functools import wraps
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import torch
import torchvision.transforms as T
from torchtyping import TensorType

from vaporize_all_humans.config import DATA_FOLDER, OUTPUT_FOLDER

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


def validate_input_images(image_paths: list[str]) -> None:
    assert len(image_paths) > 0, "the list of input images cannot be empty"
    for img in image_paths:
        assert os.path.exists(img), f"input image {img} does not exists"
        assert is_image_file(
            img
        ), f"input file {img} is not a valid image, supported formats are {IMG_EXTENSIONS}"


def get_input_images(args: Namespace) -> list[str]:
    image_paths = get_demo_images() if args.demo else args.input
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    validate_input_images(image_paths)
    return image_paths


def get_output_directory(args: Namespace, image_path: str) -> str:
    if args.demo:
        # Default directory for the demo
        output_directory = OUTPUT_FOLDER
        Path(output_directory).mkdir(exist_ok=True)
    elif args.output_directory is not None:
        # Directory specified by the user
        output_directory = args.output_directory
    else:
        # Directory of the current image
        output_directory = str(Path(image_path).parent)
    assert os.path.isdir(
        output_directory
    ), f"output directory {output_directory} does not exist"
    return output_directory


def store_output_images(args: Namespace, result: dict[str, Any]):
    if not args.disable_output_save:
        for image_path, r in result.items():
            # Where to store the image
            output_directory = get_output_directory(args, image_path)
            # Get name of the input image
            image_name, extension = os.path.splitext(os.path.basename(image_path))
            extension = extension[1:]  # Remove leading `.`
            # Add extension to the basename
            if not args.disable_output_suffix:
                image_name += args.output_suffix
            # Store the image, using the same extension as the input
            kwargs = {}
            if extension.lower() in ["jpg", "jpeg"]:
                kwargs = {"quality": 80, "optimize": True}
            T.ToPILImage()(r["predicted_image"]).save(
                os.path.join(output_directory, f"{image_name}.{extension}", **kwargs)
            )
