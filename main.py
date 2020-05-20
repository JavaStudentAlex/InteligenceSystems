from lab1.lab1 import read_dataset
from lab2.lab2 import IEIModel
import itertools

source_dir = "./images"
classes = ["wood", "cloth", "tile", "brick"]
file_pattern = "*{}*.jpg"

# get the dataset from files in pandas data frame format
dataset, features = read_dataset(source_dir, classes, file_pattern)

# split it into train and exam
train = dataset.sample(frac=0.8)
test = dataset.drop(train.index)

model = IEIModel(train, features, "class")
