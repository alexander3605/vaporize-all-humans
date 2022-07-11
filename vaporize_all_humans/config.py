import os
from argparse import ArgumentParser, Namespace

DATA_FOLDER = "data/samples"
GOLDEN_FOLDER = os.path.join(DATA_FOLDER, "golden")
OUTPUT_FOLDER = os.path.join(DATA_FOLDER, "output")
BOUNDING_BOX_MIN_SIZE = 512
BOUNDING_BOX_INCREASE_FACTOR = 0.5
IOU_FILTER_THRESHOLD = 0.5
GAUSSIAN_BLUR_ON_MERGE = True
CONDENSE_SEGMENTATION_MASKS = True


def parse_arguments() -> Namespace:
    parser = ArgumentParser(
        description="Vaporize all humans (only in images, for now).",
    )
    # Input/output arguments
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        nargs="+",
        help="Path to one or more images from which to erase humans.",
    )
    parser.add_argument(
        "-o",
        "--output-directory",
        type=str,
        help="""
            Path to the directory where the output images are stored. 
            If not present, store them in the same folder as the input images.
        """,
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="-VAPORIZED",
        help="Suffix appended to each output image.",
    )
    parser.add_argument(
        "--disable-output-suffix",
        action="store_true",
        help="""
            If present, do not append any suffix to the output image.
            Be careful, if output-directory is not present, this will 
            override the input images!
        """,
    )
    parser.add_argument(
        "--disable-output-save",
        action="store_true",
        help="If present, do not save output images.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="""
            If present, vaporize the sample images in demo/samples instead of using the selected input images. Results will be stored in demo/samples/output
            """,
    )
    # Vaporizer configuration
    parser.add_argument(
        "-s",
        "--patch-size",
        type=int,
        default=BOUNDING_BOX_MIN_SIZE,
        help="""
            Size of patches used by the vaporizer. Higher values give better quality, 
            but it takes more time to vaporize humans. Recommended values are in the range [128, 512].
            """,
    )
    parser.add_argument(
        "--patch-increase-factor",
        type=float,
        default=BOUNDING_BOX_INCREASE_FACTOR,
        help="""
            Percentage increase applied to the initial bounding boxes that locate humans in the picture.
            Higher values can give better quality, but it takes more time to vaporize humans. 
            """,
    )
    parser.add_argument(
        "--iou-filter-threshold",
        default=IOU_FILTER_THRESHOLD,
        type=float,
        help="""
            Remove inpainting patches whose Intersection-over-Union is above this threhsold.
            Higher values can make vaporization faster, and prevent stitching artifacts,
            although it might cause some humans to survive vaporization.
            """,
    )
    parser.add_argument(
        "--blur-stitching",
        action="store_true",
        help="""
             If present, apply a slight Gaussian Blur to each inpainting mask before performing the final stitching.
             It can slighlty improve the output quality, but it takes more time to vaporize humans. 
             """,
    )
    parser.add_argument(
        "--disable-patches-condensation",
        action="store_true",
        help="""
             If present, do not apply the patch condensation step in which we 
             recompute patches from the bounding boxes of disjoint segmenentation masks.
             Disabling this step can make vaporization faster, but it slightly lowers the output quality.
             """,
    )
    # Visualization settings
    parser.add_argument(
        "--show-patches",
        action="store_true",
        help="""
             If present, show all the inpainted patches, along with the original image patch and the segmentation mask. 
             """,
    )
    parser.add_argument(
        "--show-mask",
        action="store_true",
        help="If present, show the full segmentation mask for each input image, and how it has been inpainted.",
    )
    parser.add_argument(
        "--show-result",
        action="store_true",
        help="If present, show the final image where humans have been vaporized.",
    )
    return parser.parse_args()
