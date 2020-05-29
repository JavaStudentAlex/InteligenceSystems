from itertools import product
from skimage import io
import pandas as pd
import numpy as np
import fnmatch
import os


# high level function for building dataset
def read_dataset(source_dir, classes, file_pattern, standard_shape):

    source_class_gen = make_images_sources(source_dir, classes, file_pattern)
    dataset = build_dataset(source_class_gen, standard_shape)
    return dataset


# read the source directory content and return all pathes of classes images_for_labs
def make_images_sources(source_dir, classes, file_pattern):
    sources = list()
    for class_name in classes:
        full_paths_files_source_dir = (os.path.abspath("{}/{}".format(source_dir, file_name))
                                       for file_name in os.listdir(source_dir))
        class_sources = fnmatch.filter(full_paths_files_source_dir, file_pattern.format(class_name))
        sources.extend(class_sources)
        yield from product(class_sources, [class_name])


# read each image, cut it for standard size, reshape for feature vector form (1 * features)
# and insert into the DataFrame also with the class label
def build_dataset(source_class_gen, std_shape):
    columns = make_columns(std_shape) + ["class"]
    rows = list()

    for image_path, class_name in source_class_gen:
        feature_vector_length = np.prod(std_shape)
        image_matrix = io.imread(image_path)
        cut_img = cut_image(image_matrix, std_shape)
        feature_vector = cut_img.reshape(feature_vector_length)
        rows.append((*feature_vector, class_name))
    return pd.DataFrame(data=rows, columns=columns), columns[:-1]


def make_columns(std_shape):
    return ["{}:{}:{}".format(*triple) for triple in product(range(1, std_shape[0] + 1),
                                                             range(1, std_shape[1] + 1),
                                                             range(1, std_shape[2] + 1))]


# cut standard image size from the center
def cut_image(image_matrix, std_shape):
    real_shape = image_matrix.shape

    height_border = calc_border(std_shape, real_shape, 0)
    width_border = calc_border(std_shape, real_shape, 1)

    height_size = calc_size(std_shape, real_shape, 0)
    width_size = calc_size(std_shape, real_shape, 1)

    result_matrix = np.zeros(std_shape)
    result_matrix[:height_size, :width_size] = image_matrix[height_border:height_border+height_size,
                                                            width_border:width_size+width_border]
    return result_matrix


# calc the border value
def calc_border(std_vals, real_vals, axis_index):
    return only_positive_int_numbers((real_vals[axis_index] - std_vals[axis_index]) / 2)


# return positive number or 0
def only_positive_int_numbers(val):
    return int(val) if val > 0 else 0


# if the real image size param(width or height through the axis_index) is bigger than
# standard return standart value if not return real value of the image
def calc_size(std_vals, real_vals, axis_index):
    real_param = real_vals[axis_index]
    std_param = std_vals[axis_index]
    return std_param if real_param - std_param > 0 else real_param

