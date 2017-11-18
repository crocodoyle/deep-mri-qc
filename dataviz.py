
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import math
import numpy as np

import nibabel as nib

workdir = '/data1/data/deepqc/'


import imageio
import os

def gif_my_brain(input_file):
    t1_image = nib.load(input_file).get_data()
    print(t1_image.shape)

    plt.imshow(t1_image[:, int(t1_image.shape[1]/2), :].T, cmap='gray')

    plt.tight_layout()
    plt.axis('off')
    plt.savefig('E:/brains/andrew.png')

    x_range, y_range, z_range = t1_image.shape

    for y in range(y_range):
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,  hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        plt.imshow(t1_image[:, int(y), :].T, cmap='gray')
        plt.axis('off')
        plt.savefig('E:/brains/andrew/' + str(y) + '.png', bbox_inches='tight', pad_inches=0)

    start_slice = 202
    end_slice = 68

    images = []
    for y in range(start_slice, end_slice, -1):
        images.append(imageio.imread('E:/brains/andrew/' + str(y) + '.png'))

    for y in range(end_slice, start_slice):
        images.append(imageio.imread('E:/brains/andrew/' + str(y) + '.png'))

    imageio.mimsave('E:/brains/andrew/andrew.gif', images)

def plot_nonlinearities(output_path):

    x = np.linspace(-10.0, 10.0, 20000)

    relu = np.zeros(x.shape)
    sigmoidal = np.zeros(x.shape)
    tanh = np.zeros(x.shape)
    leakyReLu = np.zeros(x.shape)

    for i, x_ in enumerate(x):
        if x_ > 0:
            relu[i] = x_
            leakyReLu[i] = x_
        else:
            relu[i] = 0
            leakyReLu[i] = 0.1*x_

        sigmoidal[i] = sigmoid(x_)
        tanh[i] = np.tanh(x_)


    f, ax = plt.subplots(2, 2, sharex=True, sharey=True)

    ax[0][0].plot(x, sigmoidal)
    ax[0][0].set_title('$\sigma$')
    ax[0][0].grid()

    ax[0][1].plot(x, relu)
    ax[0][1].set_title('ReLU')
    ax[0][1].grid()

    ax[1][0].plot(x, tanh)
    ax[1][0].set_title('tanh')
    ax[1][0].grid()

    ax[1][1].plot(x, leakyReLu)
    ax[1][1].set_title('Leaky ReLU')
    ax[1][1].grid()

    # plt.tight_layout()

    plt.suptitle('Non-Linear Activations')

    ax[0][0].set_xlim([-3, 3])
    ax[0][0].set_ylim([-1.1, 1.1])
    plt.savefig(output_path + 'nonlinearities.png', bbox_inches='tight')

def sigmoid(x):
  return 1 / (1 + math.exp(-x))




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

    plot_nonlinearities('E:/')

    # gif_my_brain('E:/brains/andrew_mri_nov_2015.mnc')

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