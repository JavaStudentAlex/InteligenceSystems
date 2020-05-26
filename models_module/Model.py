import pandas as pd
import numpy as np
from models_module.Coach import Coach
from concurrent.futures import ThreadPoolExecutor


class IEIModel:

    __tolerance_dist_start = 20

    def __init__(self, train_data, features, target):
        self.__features = features
        self.__target = target
        self.__classes = {*np.unique(train_data[self.__target].values)}
        self.__train(train_data)
        self.__report()

    def __train(self, train_data):
        base_class = train_data[self.__target].mode().values[0]
        mean_base_class_val = self.__mean_base_class_values(train_data, base_class)

        features_len = len(mean_base_class_val)

        with ThreadPoolExecutor(max_workers=4) as executor:
            for feature_number in range(features_len):
                executor.submit(IEIModel.optimize_feature,
                                train_data,
                                self.__features,
                                self.__target,
                                self.__tolerance_dist_start,
                                feature_number,
                                mean_base_class_val)


    @staticmethod
    def optimize_feature(data, features, target, dist_start, feature_index, features_mean):

        tol_dist_start_vals = np.array([dist_start] * len(features))
        top_feature_val = int(features_mean[feature_index] / 2)

        max_kfe = float("-inf")
        optimal_val = None

        for tol_dist_for_feature in range(dist_start, top_feature_val + 1):
            tol_dist_start_vals[feature_index] = tol_dist_for_feature

            coach = Coach(data, features, target, features_mean, tol_dist_start_vals)

            if coach.get_overall_KFE() > max_kfe:
                optimal_val = tol_dist_for_feature

        print("for {} feature is {} optimal value",feature_index, optimal_val )
        return {features[feature_index]: optimal_val}

    def __calc_bottom_up_top(self, means, tol_radiuses):
        return means - tol_radiuses, means + tol_radiuses


    def __mean_base_class_values(self, dataset, most_freq_class):
        values_matrix = dataset.loc[dataset[self.__target] == most_freq_class, self.__features].values
        measure_number = values_matrix.shape[0]
        mean_feature_vals = np.sum(values_matrix, axis=0) / measure_number
        return mean_feature_vals

    def __report(self):
        print("Model was built with {} classes:".format(len(self.__classes)))
        for class_name in self.__classes:
            print("{} class with {} radius and {} KFE".format(class_name,
                                                              self.__coach.radiuses.loc[class_name, "radius"],
                                                              self.__coach.coeffs.loc[class_name, "KFE"]))
            print("Tol interval ditance is :\n{}".format(self.__coach.tol_dists))




        




