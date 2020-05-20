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

    def __train(self, train_data):
        most_freq_class = train_data[self.__target].mode().values[0]
        self.__cont_tol_interval = self.__build_cont_tolerance_field(train_data, most_freq_class)
        binarized_dataset = self.__build_bin_feature_matrix(train_data)
        self.__centers = self.__build_standard_bin_classes_vectors(binarized_dataset)
        neighbours = self.__define_neighbours()

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
            neighbour_name = self.__get_neighbour(class_name, other_classes)
            neighbours[class_name] = neighbour_name
        return neighbours

    def __get_neighbour(self, target_class_name, other_classes):
        this_class_center = self.__centers.loc[target_class_name]
        distances = dict()
        for current_other_class_name in other_classes:
            current_other_class_center = self.__centers.loc[current_other_class_name]
            hemming_distance = len(np.where(this_class_center != current_other_class_center)[0])
            distances[current_other_class_name] = hemming_distance
        dists_frame = pd.DataFrame.from_dict(distances, orient="index", columns=["0"])
        return dists_frame["0"].idxmin()




