from prepare_package.prepare_module import read_dataset
from models_module.IEI import IEIModelAPI
from datetime import datetime as dt

start = dt.now()
source_dir = "images_for_labs"
classes = ["wood", "cloth", "tile", "brick"]
file_pattern = "*{}*.jpg"
standard_shape = (50, 50, 3)

# get the dataset from files in pandas data frame format
dataset, features = read_dataset(source_dir, classes, file_pattern, standard_shape)

# split it into train and exam
train = dataset.sample(frac=0.8)
test = dataset.drop(train.index)

# make the model for predictions
model = IEIModelAPI(train, features, "class")

#test["predicted"] = model.predict(test)

finish = dt.now()
print(finish - start)

