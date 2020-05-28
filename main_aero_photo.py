from prepare_package.prepare_module import read_dataset
from models_module.IEI import IEIModelAPI

source_dir = "aero_photo_train"
classes = ["field", "road", "town", "water"]
file_pattern = "*{}*.jpg"
standard_shape = (100, 100, 3)

# get the dataset from files in pandas data frame format
dataset, features = read_dataset(source_dir, classes, file_pattern, standard_shape)

model = IEIModelAPI(dataset, features, "class")

print("here")
