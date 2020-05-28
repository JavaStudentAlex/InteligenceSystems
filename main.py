from prepare_package.prepare_module import read_dataset
from models_module.IEI import IEIModelAPI
from datetime import datetime as dt

start = dt.now()
source_dir = "./images"
classes = ["wood", "cloth", "tile", "brick"]
file_pattern = "*{}*.jpg"

# get the dataset from files in pandas data frame format
dataset, features = read_dataset(source_dir, classes, file_pattern)

# split it into train and exam
train = dataset.sample(frac=0.8)
test = dataset.drop(train.index)

# make the model for predictions
model = IEIModelAPI(train, features, "class")

#test["predicted"] = model.predict(test)

finish = dt.now()
print(finish - start)

