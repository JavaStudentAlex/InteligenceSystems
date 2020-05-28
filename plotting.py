from prepare_package.prepare_module import read_dataset
from matplotlib import pyplot as plt


def plot_class(dataset, class_name, style):
    class_data = dataset[dataset["class"] == class_name]
    x = class_data.iloc[:, 0].values
    y = class_data.iloc[:, 1].values
    x_mean = x.mean()
    y_mean = y.mean()
    plt.annotate("{}_center".format(class_name),
                 xy=(x_mean, y_mean), xycoords="data",
                 xytext=(x_mean+20, y_mean+20), textcoords="data",
                 arrowprops=dict(facecolor='b', shrink=0.05, width=2))
    plt.plot([*x, x_mean], [*y, y_mean], style, label=class_name)


source_dir = "images_for_labs"
classes = ["wood", "cloth", "tile", "brick"]
plotting_styles = ["ro", "g^", "b+", "bs"]
file_pattern = "*{}*.jpg"

# get the dataset from files in pandas data frame format
dataset, features = read_dataset(source_dir, classes, file_pattern)

for class_name, style in zip(classes, plotting_styles):
    plot_class(dataset, class_name, style)

plt.legend()
plt.show()




