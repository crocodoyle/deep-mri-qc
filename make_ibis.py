import numpy as np
import h5py

import os, csv, time
import nibabel as nib

# from custom_loss import sensitivity, specificity, true_positives, true_negatives, false_positives, false_negatives

from collections import defaultdict

import pickle as pkl

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import confusion_matrix

from vis.visualization import visualize_cam, overlay
from vis.utils import utils
from keras import activations

from dltk.core.io.preprocessing import normalise_zero_one, resize_image_with_crop_or_pad


workdir = '/home/users/adoyle/deepqc/IBIS/'
datadir = '/data1/users/adoyle/IBIS/'

label_file = datadir + 't1_ibis_QC_labels.csv'

target_size = (168, 256, 224)


def make_ibis_qc():
    total_subjects = 0

    with open(label_file, 'r') as labels_csv:
        qc_reader = csv.reader(labels_csv)
        next(qc_reader)

        for line in qc_reader:
            t1_filename = line[3][9:]

            try:
                t1 = nib.load(datadir + t1_filename)
                total_subjects += 1
            except:
                print('Missing', t1_filename)

    with h5py.File(workdir + 'ibis_revisited.hdf5', 'w') as f:
        f.create_dataset('ibis_t1', (total_subjects, target_size[0], target_size[1], target_size[2]), dtype='float32')
        f.create_dataset('qc_label', (total_subjects, 2), dtype='float32')
        dt = h5py.special_dtype(vlen=bytes)
        f.create_dataset('filename', (total_subjects, ), dtype=dt)

        index = 0

        indices = []
        labels = []

        with open(label_file, 'r') as labels_csv:
            qc_reader = csv.reader(labels_csv)
            next(qc_reader)

            for line in qc_reader:
                try:
                    t1_filename = line[3][9:]
                    label = line[4]

                    if 'Pass' in label:
                        pass_fail = 1
                    else:
                        pass_fail = 0

                    f['qc_label'][index] = pass_fail
                    t1_data = nib.load(datadir + t1_filename).get_data()

                    if not t1_data.shape == target_size:
                        # print('resizing from', t1_data.shape)
                        t1_data = resize_image_with_crop_or_pad(t1_data, img_size=target_size, mode='constant')

                    f['ibis_t1'][index, ...] = normalise_zero_one(t1_data)
                    f['filename'][index] = t1_filename.split('/')[-1]

                    # plt.imshow(t1_data[96, ...])
                    # plt.axis('off')
                    # plt.savefig(output_dir + t1_filename[:-4] + '.png', bbox_inches='tight', cmap='gray')

                    indices.append(index)
                    labels.append(pass_fail)

                    index += 1
                except Exception as e:
                    print('Error:', e)

        print('Total subjects we actually have:', index+1)
