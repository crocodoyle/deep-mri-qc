import numpy as np
from scipy.spatial.distance import euclidean

import os, sys, time, csv, subprocess

from dltk.core.io.preprocessing import normalise_zero_one, resize_image_with_crop_or_pad
import dltk
import h5py


import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import h5py
import sklearn

from sklearn.neighbors import KDTree

import nibabel as nib

from multiprocessing import Pool, Process

def flatten_IBIS():

    for root, dirs, files in os.walk('/data1/data'):
        for file in files:
            if '.mnc' in file and 'IBIS' in file:
                orig_filename = file.split('//')[-1]
                filename = file.split('//')[-1][4:]

                print(filename)
                print('/data1/data/IBIS/' + filename)
                os.rename(os.path.join(root, orig_filename), '/data1/data/IBIS/' + filename)


def check_hdf5():
    f = h5py.File('/data1/data/deepqc/deepqc.hdf5', 'r')
    images = f['MRI']

    for i, image in enumerate(images):
        plt.imshow(image[:, :, 100])
        plt.savefig('/data1/data/deepqc/test/' + str(i) + '.png')


if __name__ == "__main__":
    check_hdf5()