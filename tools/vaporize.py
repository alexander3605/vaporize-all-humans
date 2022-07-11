from vaporize_all_humans.config import parse_arguments
from vaporize_all_humans.utils import get_input_images, store_output_images
from vaporize_all_humans.vaporizer import Vaporizer

# TODO: handle shadows: identify each contiguous region in the mask, flip it vertically, shift it down, and extrude
# To identify regions, use the original bounding boxes. Extend them by 10% to be safe, but only left/right/top, not bottom
# TODO: rotate all boxes around bottom, starting from 60-60 angle. Look for min average brightness, that's probably the shadow angle


if __name__ == "__main__":

    # Obtain the vaporizer configuration
    args = parse_arguments()

    # Get the list of images to process
    images = get_input_images(args)

    # Create a vaporizer
    vaporizer = Vaporizer(
        patch_size=args.patch_size,
        patch_increase_factor=args.patch_increase_factor,
        iou_filter_threshold=args.iou_filter_threshold,
        blur_stitching=args.blur_stitching,
        patches_condensation=not args.disable_patches_condensation,
        show_patches=args.show_patches,
        show_mask=args.show_mask,
        show_result=args.show_result,
    )

    # Inference
    result = vaporizer(images)
    result_dataframe = vaporizer.dataframe
    print(result_dataframe)
    print(
        f"average PSNR={result_dataframe[result_dataframe['psnr'] > 0]['psnr'].mean():.2f} dB"
    )

    # Store output images
    store_output_images(args, result)
