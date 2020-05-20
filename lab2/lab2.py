from math import log2
import pandas as pd
import numpy as np



class IEIModel:

    __tolerance_dist = 20
    __sel_level = 0.5

    def __init__(self, train_data, features, target):
        self.__features = features
        self.__target = target
        self.__classes = {*np.unique(train_data[self.__target].values)}
        self.__train(train_data)
        self.__report()

    def __train(self, train_data):
        most_freq_class = train_data[self.__target].mode().values[0]
        self.__cont_tol_interval = self.__build_cont_tolerance_field(train_data, most_freq_class)
        binarized_dataset = self.__build_bin_feature_matrix(train_data)
        self.__centers = self.__build_standard_bin_classes_vectors(binarized_dataset)
        neighbours = self.__define_neighbours()
        self.__radiuses = self.__build_optimal_classes_radiuses(binarized_dataset, neighbours)

    def __build_cont_tolerance_field(self, dataset, most_freq_class):
        values_matrix = dataset.loc[dataset[self.__target] == most_freq_class, self.__features].values
        measure_number = values_matrix.shape[0]
        mean_feature_vals = np.sum(values_matrix, axis=0) / measure_number
        return mean_feature_vals - self.__tolerance_dist, mean_feature_vals + self.__tolerance_dist

    def __build_bin_feature_matrix(self, dataset):
        val_matrix = dataset[self.__features].values
        bin_val_matrix = np.zeros(shape=val_matrix.shape)

        positions = np.where((val_matrix > self.__cont_tol_interval[0]) &
                             (val_matrix < self.__cont_tol_interval[1]))
        bin_val_matrix[positions] = 1

        bin_dataset = dataset.copy()
        bin_dataset[self.__features] = bin_val_matrix

        return bin_dataset

    def __build_standard_bin_classes_vectors(self, binarized_dataset):
        centers = {}
        for class_name in self.__classes:
            class_matrix = binarized_dataset.loc[binarized_dataset[self.__target] == class_name,
                                                   self.__features].values
            cont_center = self.__calc_container_center(class_matrix)
            centers[class_name] = cont_center
        return pd.DataFrame.from_dict(centers, orient="index", columns=self.__features)

    def __calc_container_center(self, matrix):
        mean_bin_vals = np.sum(matrix, axis=0) / matrix.shape[0]
        cur_cont_center = np.zeros(matrix.shape[1])
        units_positions = np.where(mean_bin_vals > self.__sel_level)
        cur_cont_center[units_positions] = 1
        return cur_cont_center

    def __define_neighbours(self):
        neighbours = dict()
        for class_name in self.__classes:
            other_classes = self.__classes.difference({class_name})
            neighbour_name = self.__find_neighbour_for_current_class(class_name, other_classes)
            neighbours[class_name] = neighbour_name
        return neighbours

    def __find_neighbour_for_current_class(self, target_class_name, other_classes):
        this_class_center = self.__centers.loc[target_class_name]
        distances = dict()
        for current_other_class_name in other_classes:
            current_other_class_center = self.__centers.loc[current_other_class_name]
            hemming_distance = self.__calc_hemming_distance(this_class_center,
                                                            current_other_class_center)
            distances[current_other_class_name] = hemming_distance
        dists_frame = pd.DataFrame.from_dict(distances, orient="index", columns=["0"])
        return dists_frame["0"].idxmin()

    def __calc_hemming_distance(self, vector_1, vector_2):
        return len(np.where(vector_1 != vector_2)[0])

    def __build_optimal_classes_radiuses(self, bin_dataset, neighbours):
        radiuses = dict()
        for cur_class_name in self.__classes:
            nghbr_class_name = neighbours[cur_class_name]

            cur_class_center = self.__centers.loc[cur_class_name].values
            cur_class_msrs_matrix = self.__get_measures(cur_class_name, bin_dataset)
            nghbr_class_msrs_matrix = self.__get_measures(nghbr_class_name, bin_dataset)

            radiuses[cur_class_name] = self.__calc_optimal_radius(cur_class_center, cur_class_msrs_matrix,
                                                                         nghbr_class_msrs_matrix)
        return radiuses

    def __get_measures(self, class_name, bin_dataset):
        return bin_dataset.loc[bin_dataset[self.__target] == class_name, self.__features].values

    def __calc_optimal_radius(self, goal_class_center, goal_class_matrix, neighbour_class_matrix):
        cases = dict()
        feature_number = goal_class_matrix.shape[1]

        for radius in range(1, feature_number+1):
            best_func_efficiency_coof = self.__calc_inf_efficiency_coefficient(radius,
                                                                               goal_class_center,
                                                                               goal_class_matrix,
                                                                               neighbour_class_matrix)
            cases[radius] = best_func_efficiency_coof

        result_radius = pd.DataFrame.from_dict(cases, orient="index").idxmax()
        return result_radius, cases[result_radius]

    def __calc_inf_efficiency_coefficient(self, radius, goal_class_center,
                                          cur_class_matrix, nghbr_class_matrix):
        cur_in, cur_out = self.__calc_how_many_in_out_measures(radius, goal_class_center, cur_class_matrix)
        nghbr_in, nghbr_out = self.__calc_how_many_in_out_measures(radius, goal_class_center, nghbr_class_matrix)
        return self.__calc_shannon_criteria(cur_in, cur_out, nghbr_in, nghbr_out)

    def __calc_how_many_in_out_measures(self, radius, center, measures):
        in_measures = 0
        out_measures = 0

        for measure in measures:
            hemming_dist = self.__calc_hemming_distance(measure, center)
            if hemming_dist <= radius:
                in_measures += 1
            else:
                out_measures += 1
        return in_measures, out_measures

    def __calc_shannon_criteria(self, k1, k2, k3, k4):
        def n_log_n(k, p):
            n = k / (k + p)
            return n * log2(n)
        if k1 == 0 or k4 == 0:
            return 0

        return 1 + 1/2 * (n_log_n(k1, k3) + n_log_n(k2, k4) + n_log_n(k3, k1) + n_log_n(k4, k2))

    def __report(self):
        print("Model was built with {} :".format(len(self.__classes)))
        for class_name in self.__classes:
            print("{} class with {} radius".format(class_name, self.__radiuses[class_name]))



        




