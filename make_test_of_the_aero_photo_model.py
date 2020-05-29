import pickle as pkl
from prepare_package.prepare_module import make_columns
from split_image import split_the_image_in_boxes
import pandas as pd
from datetime import datetime as dt
from skimage.io import imsave
import numpy as np

start = dt.now()


def make_frame_of_images_matrices(matrices, features_columns):
    features_len = len(features_columns)

    # container for collecting all feature vectors
    index_measure_dict = dict()

    for i in range(matrices.shape[0]):
        feature_vector = matrices[i].reshape(features_len)
        index_measure_dict[i] = feature_vector

    return pd.DataFrame.from_dict(index_measure_dict, orient="index", columns=features_columns)


# path of the aero photo image
aero_photo_image = "aerophoto.jpg"

# standard size of the small images
std_size = (50, 50, 3)

# make columns names for dataset
columns = make_columns(std_size)

# cut the big image
small_images = split_the_image_in_boxes(aero_photo_image, std_size)

# make the data frame for testing
test_dataset = make_frame_of_images_matrices(small_images, columns)

# free memory
del small_images

# path to model
model_path = "model_bin/model.pkl"

with open(model_path, "rb") as model_file:
    model = pkl.load(model_file)

# create column inside dataset with predicted results
test_dataset["predicted"] = model.predict(test_dataset)

for index in test_dataset.index:
    # define the name of the image
    path_where_to_save = "aero_photo_predict/{}_{}.jpg".format(index, test_dataset.loc[index, "predicted"])

    # convert to the std size from feature vector size and save
    imsave(path_where_to_save, np.array(test_dataset.loc[index, columns].values, dtype="int").reshape(std_size))
