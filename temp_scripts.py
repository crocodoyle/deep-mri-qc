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

def generate_iseg_images():

    iseg_dir = 'E:/iseg2017/'

    t1 = nib.load(iseg_dir + 'training/subject-1-T1.img').get_data()
    t2 = nib.load(iseg_dir + 'training/subject-1-T2.img').get_data()
    labels = nib.load(iseg_dir + 'training/subject-1-label.img').get_data()

    print(t1.shape, labels.shape)


    plt.imshow(t1[:, :, t1.shape[2]//2+10, 0].T, cmap='gray')
    plt.axis('off')
    plt.savefig(iseg_dir + 'example_t1.png', bbox_inches='tight')

    plt.imshow(t2[:, :, t2.shape[2]//2+10, 0].T, cmap='gray')
    plt.axis('off')
    plt.savefig(iseg_dir + 'example_t2.png', bbox_inches='tight')

    plt.imshow(labels[:, :, labels.shape[2]//2+10, 0].T)
    plt.axis('off')
    plt.savefig(iseg_dir + 'example_labels.png', bbox_inches='tight')


    #IBIS 3 timepoints
    ibis_dir = 'E:/iseg2017/IBIS/103430/'

    t1_06 = nib.load(iseg_dir + 'IBIS/103430/V06/deface/ibis_103430_V06_t1w.mnc').get_data()
    t1_12 = nib.load(ibis_dir + 'V12/deface/deface_103430_V12_t1w.mnc').get_data()
    t1_24= nib.load(ibis_dir + 'V24/deface/deface_103430_V24_t1w.mnc').get_data()

    labels1 = nib.load(iseg_dir + 'IBIS/103430_V06_label.nii.gz').get_data()
    labels2 = nib.load(iseg_dir + 'IBIS/107524_V06_label.nii.gz').get_data()


    print(t1_06.shape)
    print(t1_12.shape)


    plt.imshow(t1_06[:, t1_06.shape[1]//2-10, :].T, cmap='gray')
    plt.axis('off')
    plt.savefig(iseg_dir + 'ibis_v06_example.png')

    plt.imshow(t1_12[:, t1_12.shape[1]//2, :].T, cmap='gray')
    plt.axis('off')
    plt.savefig(iseg_dir + 'ibis_v12_example.png')

    plt.imshow(t1_24[:, t1_24.shape[1]//2, :].T, cmap='gray', origin='lower')
    plt.axis('off')
    plt.savefig(iseg_dir + 'ibis_v24_example.png')

    plt.imshow(labels1[:, :, labels1.shape[2]//2].T, origin='lower')
    plt.axis('off')
    plt.savefig(iseg_dir + 'ibis_v06_example_labels.png', bbox_inches='tight')

    plt.imshow(labels2[:, :, labels2.shape[2]//2+10].T, origin='lower')
    plt.axis('off')
    plt.savefig(iseg_dir + 'ibis_v24_example_labels.png', bbox_inches='tight')


if __name__ == "__main__":
    # check_hdf5()

    generate_iseg_images()