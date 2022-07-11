from vaporize_all_humans.vaporizer import Vaporizer
from vaporize_all_humans.utils import is_image_file
import os
from vaporize_all_humans.config import DATA_FOLDER

# TODO: handle shadows: identify each contiguous region in the mask, flip it vertically, shift it down, and extrude
# To identify regions, use the original bounding boxes. Extend them by 10% to be safe, but only left/right/top, not bottom
# TODO: rotate all boxes around bottom, starting from 60-60 angle. Look for min average brightness, that's probably the shadow angle


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
