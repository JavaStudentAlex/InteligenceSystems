from prepare_package.prepare_module import read_dataset
from models_module.IEI import IEIModelAPI
import pickle as pkl

# where small images situated
source_dir = "aero_photo_train"

# properties for extracting dataset
classes = ["field", "road", "town", "water"]
file_pattern = "*{}*.jpg"
standard_shape = (50, 50, 3)

# where save the model
target_dir_for_bin_model = "model_bin"

# get the dataset from files in pandas data frame format
dataset, features = read_dataset(source_dir, classes, file_pattern, standard_shape)

model = IEIModelAPI(dataset, features, "class")

# serialize the model
with open("{}/{}.pkl".format(target_dir_for_bin_model, "model"), "wb") as bin_model_file:
    pkl.dump(model, bin_model_file, protocol=pkl.HIGHEST_PROTOCOL)

