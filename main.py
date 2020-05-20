from lab1.lab1 import read_dataset
import itertools

source_dir = "./images"
classes = ["wood", "cloth", "tile", "brick"]
file_pattern = "*{}*.jpg"

# get the dataset from file in pandas data frame
dataset = read_dataset(source_dir, classes, file_pattern)

# split it into train and exam
train = dataset.sample(frac=0.8)
test = dataset.drop(train.index)
print(test['class'])
