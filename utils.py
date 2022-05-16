import imageio
import os


def draw_gif(image_dir, gif_name):
    image_fs = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            image_fs.append(os.path.join(root, file))
    with imageio.get_writer(gif_name, mode='I') as writer:
        print('Generate Predicted GIF')
        for filename in image_fs:
            image = imageio.v3.imread(filename)
            writer.append_data(image)
            print(filename + ' added.')