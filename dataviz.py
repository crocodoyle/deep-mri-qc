import h5py

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


import nibabel as nib

workdir = '/data1/data/deepqc/'


from vis.utils import utils
from keras import activations


import imageio
import os

def gif_my_brain(input_file):
    t1_image = nib.load(input_file).get_data()
    print(t1_image.shape)

    plt.imshow(t1_image[:, int(t1_image.shape[1]/2), :], cmap='gray')

    plt.tight_layout()
    plt.axis('off')
    plt.savefig('E:/brains/andrew.png')


    x_range, y_range, z_range = t1_image.shape

    for y in range(y_range):
        plt.imshow(t1_image[:, int(y), :].T, cmap='gray')
        plt.axis('off')
        plt.savefig('E:/brains/andrew/' + str(y) + '.png', bbox_inches='tight')

    start_slice = 202
    end_slice = 68

    images = []
    for y in range(start_slice, end_slice, -1):
        images.append(imageio.imread('E:/brains/andrew/' + str(y) + '.png'))

    for y in range(end_slice, start_slice):
        images.append(imageio.imread('E:/brains/andrew/' + str(y) + '.png'))

    imageio.mimsave('E:/brains/andrew/andrew.gif', images)


def rename_abide(input_path, output_path):

    for file in os.listdir(input_path):
        print(file)
        tokens = file.split('+')
        id = tokens[1]

        print(id)
        os.rename(input_path + file, output_path + id[2:] + '.mnc')

def visualize_fail_regions():
    pass


if __name__ == '__main__':


    gif_my_brain('E:/brains/andrew_mri_nov_2015.mnc')

    # rename_abide('E:/abide1/natives/', 'E:/abide1/abide/')

    # f = h5py.File(workdir + 'deepqc.hdf5')
    #
    # images = f['MRI']
    #
    # for i, image in enumerate(images):
    #     filename = workdir + str(i) + '.png'
    #
    #     plt.imshow(image[96, ...])
    #     plt.savefig(filename)