from skimage import io

parent_dir = "./images"
image_name = "aerophoto.jpg"
standard_size = (50, 50, 3)
all_pixels = io.imread("{}/{}".format(parent_dir, image_name))

def grid_image(image_pixels, standart_grid_size):
    pass
