import os
import glob
import h5py

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def _calculate_strain(dx, dX):
    F = np.linalg.solve(dX, dx)
    C = F.T @ F
    E = 0.5 * (C - np.identity(3))
    return E


def _load_file(filename):
    with h5py.File(filename, 'r') as f:
        tracked_points = list(f.items())[1][1].value

    return tracked_points


def _get_images(im_path):
    im_list = list()
    for filename in os.listdir(im_path):
        im_list.append(os.path.join(im_path, filename))

    return im_list


def _plot_points_on_image(images, points):
    fig, ax = plt.subplots()
    output_path = 'D:\\sparc\\ucla_pig_heart\\rig_test\\v3\\tracking_images'

    jet = plt.get_cmap('YlOrBr')
    colors = iter(jet(np.linspace(0, 1, len(images))))

    for index in range(len(images)):
        # image plot
        img = plt.imread(images[index])
        plt.imshow(img)

        plt.scatter(points[index, 0, :], points[index, 1, :], c='r', s=2)

        output_im = os.path.basename(images[index])
        plt.savefig(os.path.join(output_path, output_im))
        fig.canvas.draw_idle()
        plt.pause(0.4)
        plt.clf()

    plt.show()
    _create_gif(output_path)


def _create_gif(im_paths):
    fp_in = 'D:\\sparc\\ucla_pig_heart\\rig_test\\v3\\tracking_images\\*.png'
    output_gif = "D:\\sparc\\ucla_pig_heart\\rig_test\\v3\\tracked.gif"

    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=output_gif, format='GIF', append_images=imgs,
             save_all=True, duration=200, loop=0)


if __name__ == '__main__':

    root_dir = 'D:\\sparc\\ucla_pig_heart\\rig_test'

    h5_filename = 'v3\\heart_tracking_results.h5'
    h5_file_path = os.path.join(root_dir, h5_filename)

    # load hdf file and get tracked data
    tracked_points = _load_file(h5_file_path)

    image_dir = 'v3\\images'
    images_path = os.path.join(root_dir, image_dir)

    images = _get_images(images_path)

    _plot_points_on_image(images, tracked_points)

    print('done')
