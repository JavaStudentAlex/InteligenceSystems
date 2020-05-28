from prepare_package.prepare_module import read_dataset
from models_module.IEI import IEIModelAPI
from datetime import datetime as dt
import pickle as pkl

start = dt.now()

# where small images situated
source_dir = "aero_photo_train"

# properties for extracting dataset
classes = ["field", "road", "town", "water"]
file_pattern = "*{}*.jpg"
standard_shape = (100, 100, 3)

# where save the model
target_dir_for_bin_model = "model_bin"

# get the dataset from files in pandas data frame format
dataset, features = read_dataset(source_dir, classes, file_pattern, standard_shape)

model = IEIModelAPI(dataset, features, "class")

# serialize the model
with open("{}/{}.pkl".format(target_dir_for_bin_model, "model")) as bin_model_file:
    pkl.dump(model, bin_model_file)

finish = dt.now()

print(finish - start)
