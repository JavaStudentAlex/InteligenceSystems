from lab1.lab1 import main_lab1
import numpy as np
import sys


tolerance_distance = 20
np.set_printoptions(threshold=sys.maxsize)


def main_lab2():
    main_class = "wood"
    materials_dataset = main_lab1()
    tolerance_field = build_cont_tolerance_field(materials_dataset, main_class)
    bin_matrices = build_bin_matrices(materials_dataset, *tolerance_field)
    return bin_matrices


def build_cont_tolerance_field(input_dataset, main_class):
    main_cont_matrix = input_dataset[main_class]
    measure_number = main_cont_matrix.shape[0]
    mean_feature_vals = np.sum(main_cont_matrix, axis=0) / measure_number
    return mean_feature_vals - tolerance_distance, mean_feature_vals + tolerance_distance


def build_bin_matrices(measure_matrices, min_tol_field, max_tol_field):
    result_bin_matrices = {}
    for container_name, measure_matrix in measure_matrices.items():
        bin_matrix = np.zeros(measure_matrix.shape)
        positions = np.where((measure_matrix > min_tol_field) & (measure_matrix < max_tol_field))
        bin_matrix[positions] = 1
        result_bin_matrices[container_name] = bin_matrix
    return result_bin_matrices

