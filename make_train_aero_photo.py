from split_image import split_the_image_in_boxes
from skimage import io
from datetime import datetime as dt

parent_dir = "aero_photo_train"
image_name = "aerophoto.jpg"
std_size = (50, 50, 3)

image_boxes = split_the_image_in_boxes(image_name, std_size)

for small_image in image_boxes:
    path_to_save = "{}/{}.jpg".format(parent_dir, dt.now())
    io.imsave(path_to_save, small_image)


