from lab2.lab2 import main_lab2
import numpy as np

sel_level = 0.5


def main_lab3():
    bin_matrices = main_lab2()
    bin_cont_centers = calc_cont_center(bin_matrices)

    dist_inside = calc_dist_inside_container(bin_matrices, bin_cont_centers)
    center_dists = get_classes_center_dists(bin_cont_centers)
    center_measure_dists = get_dists_center_to_other_container(bin_cont_centers, bin_matrices)

    print(dist_inside)
    print(center_dists)
    print(center_measure_dists)


def calc_cont_center(bin_matrices):
    cont_centers = {}

    for cont_name, matrix in bin_matrices.items():
        mean_bin_vals = np.sum(matrix, axis=0) / matrix.shape[0]
        cur_cont_center = np.zeros(matrix.shape[1])
        units_positions = np.where(mean_bin_vals > sel_level)
        cur_cont_center[units_positions] = 1
        cont_centers[cont_name] = cur_cont_center

    return cont_centers


def calc_dist_inside_container(bin_containers, centers):
    result_dists = dict()
    for cont_name, bin_matrix in bin_containers.items():
        cur_center = centers[cont_name]

        cur_dists = calc_distances_between_center_and_measures(cur_center, cont_name, bin_matrix)

        result_dists.update(cur_dists)
    return result_dists


def calc_distances_between_center_and_measures(cur_center, cur_name, bin_matrix):
    diff_indices = np.where(bin_matrix != cur_center)
    measures_with_dist, counts = np.unique(diff_indices[0], return_counts=True)
    cur_dists = dict(zip(measures_with_dist, counts))
    measures_with_0_dists = get_measures_with_0_hemming_dist(range(bin_matrix.shape[0]), measures_with_dist)
    cur_dists.update(measures_with_0_dists)
    return {cur_name: cur_dists}


def get_measures_with_0_hemming_dist(measures, not_0_measures):
    measures = set(measures)
    not_0_measures = set(not_0_measures)
    zero_hemming_dist_measures = measures.difference(not_0_measures)
    return {zero_val_measure: 0 for zero_val_measure in zero_hemming_dist_measures}


def get_classes_center_dists(centers):
    result_dists = dict()
    cont_names = set(centers.keys())

    for curr_cont_name in cont_names:
        other_conts = cont_names.difference({curr_cont_name})
        dists_to_add = calc_center_dists_for_current_cont(curr_cont_name, other_conts, centers)
        result_dists.update(dists_to_add)
    return result_dists


def calc_center_dists_for_current_cont(current_cont_name, conts_where_to_calc, centers):
    cur_dists = dict()
    curr_cont_center = centers[current_cont_name]
    for goal_name in conts_where_to_calc:
        goal_center = centers[goal_name]
        diff = len(np.where(curr_cont_center != goal_center)[0])
        cur_dists["{} - {}".format(current_cont_name, goal_name)] = diff
    return cur_dists


def get_dists_center_to_other_container(centers, bin_matrices):
    result = dict()
    cont_names = set(centers.keys())

    for curr_cont_name in cont_names:
        other_conts = cont_names.difference({curr_cont_name})
        curr_center = centers[curr_cont_name]

        cont_result = dict()
        for gole_cont_name in other_conts:
            gole_bin_matrix = bin_matrices[gole_cont_name]
            dists = calc_distances_between_center_and_measures(curr_center, gole_cont_name, gole_bin_matrix)
            cont_result.update(dists)
        result[curr_cont_name] = cont_result

    return result




main_lab3()



