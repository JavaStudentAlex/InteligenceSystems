import numpy as np
from models_module.Model import Model
from matplotlib import pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd


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

        # get top border for delta optimizing process
        top_delta_border = int(mean_base_class_val.mean())

        # if mean top delta less than start delta than
        # use only start delta and start delta + 1 values for optimization
        if top_delta_border <= self.__tolerance_dist_start:
            top_delta_border = self.__tolerance_dist_start+1

        # number of cpu
        chunks = 4

        # prepare data for parallel processing
        generators = self.__split_deltas_range(self.__tolerance_dist_start, top_delta_border+1, chunks)
        futures = list()
        delta_kfe_pairs = dict()

        # make parallel processing
        with ThreadPoolExecutor(max_workers=chunks) as executor:
            for delta_gen in generators:
                futures.append(executor.submit(self.__parallel_delta_optimising,
                                               train_data, mean_base_class_val, delta_gen))

            # get together the parallel processing results
            for result in as_completed(futures):
                delta_kfe_pairs.update(result.result())

        # format the result data of processing
        deltas_kfe_frame = pd.DataFrame.from_dict(delta_kfe_pairs, orient="index", columns=["KFE"]).sort_index()
        best_delta = deltas_kfe_frame["KFE"].idxmax()

        # construct the final model through the optimal delta
        self.__model = self.__create_model(train_data, mean_base_class_val, best_delta, with_report=True)

        print("Model builded with {} delta and {} KFE ".format(best_delta, self.__model.get_overall_KFE()))

        plt.plot(deltas_kfe_frame.index, deltas_kfe_frame["KFE"].values)
        plt.title("Dependence of KFE from the delta value")
        plt.xlabel("delta")
        plt.ylabel("KFE")
        plt.show()

    def __mean_base_class_values(self, dataset, most_freq_class):
        # get base class feature values from dataset
        values_matrix = dataset.loc[dataset[self.__target] == most_freq_class, self.__features].values

        # get number of rows
        measure_number = values_matrix.shape[0]
        mean_feature_vals = np.sum(values_matrix, axis=0) / measure_number
        return mean_feature_vals

    # split the whole delta range for parallel processing
    def __split_deltas_range(self, start, finish, chunks):
        list_of_chanks = list()
        chunk_step = int((finish - start) / chunks)

        cur_start = start
        cur_finish = cur_start + chunk_step

        while cur_start < finish:
            if cur_finish + chunk_step - 1 >= finish:
                cur_finish = finish
            gen = range(cur_start, cur_finish)

            list_of_chanks.append(gen)

            cur_start = cur_finish
            cur_finish += chunk_step
        return list_of_chanks

    # main data for the parallel processing
    def __parallel_delta_optimising(self, data, mean_base_class_vals, deltas_generator):
        delta_kfe = dict()

        for cur_delta in deltas_generator:

            # create model for current delta abd get KFE
            model = self.__create_model(data, mean_base_class_vals, cur_delta)
            kfe = model.get_overall_KFE()
            delta_kfe[cur_delta] = kfe
        return delta_kfe

    def __create_model(self, data, mean_base_class_vals, delta, with_report=False):
        features_len = len(mean_base_class_vals)
        deltas_vector = np.array([delta] * features_len)
        return Model(data, self.__features, self.__target, mean_base_class_vals, deltas_vector, with_report)

    def predict(self, data):
        return self.__model.classify(data)


        




