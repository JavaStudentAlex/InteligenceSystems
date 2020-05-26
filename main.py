from prepare_package.prepare_module import read_dataset
from models_module.Model import IEIModel
from datetime import datetime as dt

start = dt.now()
source_dir = "./images"
classes = ["wood", "cloth", "tile", "brick"]
file_pattern = "*{}*.jpg"

# get the dataset from files in pandas data frame format
dataset, features = read_dataset(source_dir, classes, file_pattern)
print(features)

# split it into train and exam
#train = dataset.sample(frac=0.8)
#test = dataset.drop(train.index)

#model = IEIModel(train, features, "class")
#finish = dt.now()
#print(start - finish)
