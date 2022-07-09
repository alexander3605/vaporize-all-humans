from typing import TypeVar


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
