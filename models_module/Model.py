import numpy as np
import pandas as pd
from math import log2


class Model:
    __sel_level = 0.5

    def __init__(self, train_data, features, target, mean_base_class_vals, tol_interval_delta):
        # save key properties
        self.__features = features
        self.__target = target
        self.__classes = {*np.unique(train_data[self.__target].values)}

        # calc tol interval
        self.tol_dist_field_bottom_vals = mean_base_class_vals - tol_interval_delta
        self.tol_dist_field_top_vals = mean_base_class_vals + tol_interval_delta
        self.tol_dists = tol_interval_delta

        # start building the model
        self.__learn(train_data)

    def __learn(self, train_data):
        binarized_dataset = self.__build_bin_feature_matrix(train_data)
        self.__centers = self.__build_standard_bin_classes_vectors(binarized_dataset)
        neighbours = self.__define_neighbours()
        self.__radiuses, self.__coefs = self.__build_optimal_classes_radiuses(binarized_dataset, neighbours)

    def __build_bin_feature_matrix(self, dataset):
        val_matrix = dataset[self.__features].values

        # make new bin matrix
        bin_val_matrix = np.zeros(shape=val_matrix.shape)

        # assert which numbers are in tolerance interval
        positions = np.where((val_matrix > self.tol_dist_field_bottom_vals) &
                             (val_matrix < self.tol_dist_field_top_vals))

        # set 1 to numbers that are in tolerance interval
        bin_val_matrix[positions] = 1

        # create new DataFrame but with binary data
        bin_dataset = dataset.copy()
        bin_dataset[self.__features] = bin_val_matrix
        return bin_dataset

    def __build_standard_bin_classes_vectors(self, binarized_dataset):
        centers = {}
        for class_name in self.__classes:

            # get bin data for each class
            class_matrix = binarized_dataset.loc[binarized_dataset[self.__target] == class_name, self.__features].values

            # calc the container center
            cont_center = self.__calc_container_center(class_matrix)

            # add to all centers
            centers[class_name] = cont_center

        # make the DataFrame of centers
        return pd.DataFrame.from_dict(centers, orient="index", columns=self.__features)

    def __calc_container_center(self, matrix):
        # calc mean bin value for each feature
        mean_bin_vals = np.sum(matrix, axis=0) / matrix.shape[0]

        # create new center array
        cur_cont_center = np.zeros(matrix.shape[1])

        # find where mean values are bigger than selection level
        units_positions = np.where(mean_bin_vals > self.__sel_level)

        # set 1 to that positions
        cur_cont_center[units_positions] = 1
        return cur_cont_center

    def __define_neighbours(self):
        neighbours = dict()

        # define neighbour for each class
        for class_name in self.__classes:

            # define classes set without current class
            other_classes = self.__classes.difference({class_name})

            # get the neighbour
            neighbour_name = self.__find_neighbour_for_current_class(class_name, other_classes)

            # save the neighbour
            neighbours[class_name] = neighbour_name
        return neighbours

    def __find_neighbour_for_current_class(self, target_class_name, other_classes):
        this_class_center = self.__centers.loc[target_class_name]
        distances = dict()

        # we have current class and than go for each other class and calc
        # hemming distance.
        for current_other_class_name in other_classes:
            current_other_class_center = self.__centers.loc[current_other_class_name]
            hemming_distance = self.__calc_hemming_distance(this_class_center,
                                                            current_other_class_center)

            # save the distances
            distances[current_other_class_name] = hemming_distance
        dists_frame = pd.DataFrame.from_dict(distances, orient="index", columns=["0"])

        # class with the less distance will be the neighbour
        return dists_frame["0"].idxmin()

    def __calc_hemming_distance(self, vector_1, vector_2):
        return len(np.where(vector_1 != vector_2)[0])

    def __build_optimal_classes_radiuses(self, bin_dataset, neighbours):
        # containers for radiuses and KFE
        radiuses = dict()
        func_eff_coefs = dict()

        for cur_class_name in self.__classes:
            nghbr_class_name = neighbours[cur_class_name]

            cur_class_center = self.__centers.loc[cur_class_name].values
            cur_class_msrs_matrix = self.__get_measures(cur_class_name, bin_dataset)
            nghbr_class_msrs_matrix = self.__get_measures(nghbr_class_name, bin_dataset)

            radiuses[cur_class_name], func_eff_coefs[cur_class_name] = self.__calc_optimal_radius(cur_class_center,
                                                                                                  cur_class_msrs_matrix,
                                                                                                  nghbr_class_msrs_matrix)

        return pd.DataFrame.from_dict(radiuses, orient="index", columns=["radius"]), \
               pd.DataFrame.from_dict(func_eff_coefs, orient="index", columns=["KFE"])

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

        result_radius = pd.DataFrame.from_dict(cases, orient="index").idxmax().values[0]
        return result_radius, cases[result_radius]

    def __calc_inf_efficiency_coefficient(self, radius, goal_class_center,
                                          cur_class_matrix, nghbr_class_matrix):
        # calc k1, k2
        cur_in, cur_out = self.__calc_how_many_in_out_measures(radius, goal_class_center, cur_class_matrix)

        # calc k3, k4
        nghbr_in, nghbr_out = self.__calc_how_many_in_out_measures(radius, goal_class_center, nghbr_class_matrix)

        alpha = cur_out/(cur_in + cur_out)
        beta = nghbr_in/(nghbr_in + nghbr_out)
        KFE_criteria = self.__calc_KFE_criteria(alpha, beta)

        return KFE_criteria

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

    def __calc_KFE_criteria(self, alpha, beta):
        r = -10
        criteria_result = log2((2 + pow(10, r) - alpha - beta) / (alpha + beta + pow(10, r))) * (1 - alpha - beta)
        return criteria_result

    def get_overall_KFE(self):
        return self.__coefs["KFE"].mean()

    def classify(self, dataset):
        predicted_classes = list()

        bin_data = self.__build_bin_feature_matrix(dataset)

        for data_index in bin_data.index:
            # get feature vector of current measure
            feature_vector = bin_data.loc[data_index, self.__features].values

            # all values of beloning class function
            belong_frame = self.__define_class(feature_vector)

            # define what class is it
            max_belong_val = belong_frame["btc"].max()
            if max_belong_val < 0:
                predicted_classes.append("no class")
            else:
                class_name = belong_frame["btc"].idxmax()
                predicted_classes.append(class_name)
        return predicted_classes

    def __define_class(self, feature_vector):
        class_belong_to_class = dict()
        for cur_class in self.__classes:
            # calc current radius and center
            cur_class_center = self.__centers.loc[cur_class]
            cur_class_radius = self.__radiuses.loc[cur_class, "radius"]

            distance = self.__calc_hemming_distance(cur_class_center, feature_vector)

            # calc and add class beloning function
            class_beloning_function = 1 - distance/cur_class_radius
            class_belong_to_class[cur_class] = class_beloning_function

        return pd.DataFrame.from_dict(class_belong_to_class, orient="index", columns=["btc"])



