from skimage import io
import os
import fnmatch
import numpy as np


def main_lab1():
    classes = ["wood", "cloth", "tile", "brick"]
    source = "../images"
    return build_input_dataset(classes, source)


def build_input_dataset(classes, source_dir):
    result_dataset = {}
    standard_shape = (50, 50, 3)
    for class_name in classes:
        full_paths_files_source_dir = (os.path.abspath("{}/{}".format(source_dir, file_name))
                                       for file_name in os.listdir(source_dir))
        sources = fnmatch.filter(full_paths_files_source_dir, "*{}*.jpg".format(class_name))
        result_dataset[class_name] = build_container_matrix(sources, standard_shape)
    return result_dataset


def build_container_matrix(images, std_shape):
    feature_vector_length = np.prod(std_shape)
    container_matrix_shape = (len(images), feature_vector_length)
    matrix = np.zeros(container_matrix_shape)

    for image_index in range(len(images)):
        image_name = images[image_index]
        image_matrix = io.imread(image_name)
        cut_img = cut_image(image_matrix, std_shape)
        feature_vector = cut_img.reshape(feature_vector_length)
        matrix[image_index] = feature_vector
    return matrix


#cut image from the center
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


def calc_border(std_vals, real_vals, axis_index):
    return only_positive_int_numbers((real_vals[axis_index] - std_vals[axis_index]) / 2)


def only_positive_int_numbers(val):
    return int(val) if val > 0 else 0


def calc_size(std_vals, real_vals, axis_index):
    real_param = real_vals[axis_index]
    std_param = std_vals[axis_index]
    return std_param if real_param - std_param > 0 else real_param
