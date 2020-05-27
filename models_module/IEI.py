import numpy as np
from models_module.Model import Model


class IEIModelAPI:
    __tolerance_dist_start = 20

    # main constructor
    def __init__(self, train_data, features, target):
        # save all key properties
        self.__features = features
        self.__target = target
        self.__classes = {*np.unique(train_data[self.__target].values)}

        # start training
        self.__train(train_data)

    def __train(self, train_data):
        # define the most common class
        base_class = train_data[self.__target].mode().values[0]

        # calc mean feature values for base class
        mean_base_class_val = self.__mean_base_class_values(train_data, base_class)

        # make the deltas array for the features
        features_len = len(mean_base_class_val)
        features_deltas = np.array([self.__tolerance_dist_start] * features_len)
        self.__model = Model(train_data, self.__features, self.__target, mean_base_class_val, features_deltas)

    def __mean_base_class_values(self, dataset, most_freq_class):
        # get base class feature values from dataset
        values_matrix = dataset.loc[dataset[self.__target] == most_freq_class, self.__features].values

        # get number of rows
        measure_number = values_matrix.shape[0]
        mean_feature_vals = np.sum(values_matrix, axis=0) / measure_number
        return mean_feature_vals

    def predict(self, data):
        return self.__model.classify(data)


        




