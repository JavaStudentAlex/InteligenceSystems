from skimage import io
from skimage.util.shape import view_as_blocks
import numpy as np


def split_the_image_in_boxes(image_path, standard_size):
    standard_high, standard_width, number_of_colors = standard_size

    # read the image
    all_pixels = io.imread(image_path)

    # calc params for extract the compatibility image
    real_height = all_pixels.shape[0]
    real_width = all_pixels.shape[1]
    compatible_height_border = standard_high * (real_height // standard_high)
    compatible_width_border = standard_width * (real_width // standard_width)

    # split the image and divide it into blocks
    compatible_image = all_pixels[:compatible_height_border, :compatible_width_border, :]
    image_blocks_matrix = view_as_blocks(compatible_image, standard_size)

    # reshape for comfortable processing
    rows, cols = image_blocks_matrix.shape[:2]
    image_blocks_list = np.reshape(image_blocks_matrix, newshape=(rows * cols, *standard_size))

    return image_blocks_list


