import numpy as np
import h5py

import os, csv, time
import nibabel as nib

from collections import defaultdict

import pickle as pkl

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from make_datasets import normalise_zero_one, resize_image_with_crop_or_pad

data_dir = '/data1/users/adoyle/IBIS/'

label_file = data_dir + 't1_ibis_QC_labels.csv'

target_size = (160, 256, 224)


def make_ibis_qc():
    data_points = []

    with open(label_file, 'r') as labels_csv:
        qc_reader = csv.reader(labels_csv)
        next(qc_reader)

        lines = list(qc_reader)

        for line in lines:
            data_point = {}
            data_point['candidate_id'] = line[0]
            data_point['visit_label'] = line[1]
            data_point['t1_filename'] = line[3][9:]
            data_point['qc_label'] = line[4]

            try:
                t1 = nib.load(data_dir + data_point['t1_filename'])
                data_points.append(data_point)
            except:
                print('Missing', data_point['t1_filename'])

    total_subjects = len(data_points)

    with h5py.File(data_dir + 'IBIS_QC.hdf5', 'w') as f:
        f.create_dataset('MRI', (total_subjects, target_size[0], target_size[1], target_size[2], 1), dtype='float32')
        f.create_dataset('qc_label', (total_subjects,), dtype='float32')
        dt = h5py.special_dtype(vlen=bytes)
        f.create_dataset('filename', (total_subjects,), dtype=dt)
        f.swmr_mode = True

        for i, data_point in enumerate(data_points):
            if 'Pass' in data_point['qc_label']:
                pass_fail = 1
            else:
                pass_fail = 0

            f['qc_label'][i] = pass_fail
            t1_data = nib.load(data_dir + data_point['t1_filename']).get_data()

            if not t1_data.shape == target_size:
                # print('resizing from', t1_data.shape)
                t1_data = resize_image_with_crop_or_pad(t1_data, img_size=target_size, mode='constant')

            f['MRI'][i, ...] = np.reshape(normalise_zero_one(t1_data), (target_size) + (1,))
            f['filename'][i] = data_point['t1_filename']

            print(str(i+1), 'of', total_subjects, data_point['candidate_id'])

if __name__ == '__main__':
    print('Creating IBIS HDF5 file for quality control training')
    make_ibis_qc()
    print('Done!')