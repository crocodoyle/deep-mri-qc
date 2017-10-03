import numpy as np
from scipy.spatial.distance import euclidean

import os, sys, time, csv, subprocess

from dltk.core.io.preprocessing import normalise_zero_one, resize_image_with_crop_or_pad
import dltk


import h5py
import sklearn

from sklearn.neighbors import KDTree

import nibabel as nib

from multiprocessing import Pool, Process

def flatten_IBIS():
    with open(os.path.join('/data1/data/IBIS', 't1_qc.csv'), 'w') as label_file:
        qc_writer = csv.writer(label_file)

        for root, dirs, files in os.walk('/data1/data/IBIS/T1_Minc_Fail', topdown=False):
            for file in files:
                if '.mnc' in file:
                    filename = file.split('//')[-1]
                    qc_writer.writerow([filename, '0'])

        for root, dirs, files in os.walk('/data1/data/IBIS/T1_Minc_Pass', topdown=False):
            for file in files:
                if '.mnc' in file:
                    filename = file.split('//')[-1]
                    qc_writer.writerow([filename, '2'])




if __name__ == "__main__":
    flatten_IBIS()