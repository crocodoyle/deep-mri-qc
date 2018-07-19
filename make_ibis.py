import numpy as np
import h5py

import os, csv, time
import nibabel as nib

from collections import defaultdict
from skimage.exposure import adjust_log

import pickle as pkl

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import subprocess

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
        f.create_dataset('MRI', (total_subjects, 1, target_size[0], target_size[1], target_size[2]), dtype='float32')
        f.create_dataset('qc_label', (total_subjects,), dtype='float32')
        dt = h5py.special_dtype(vlen=bytes)
        f.create_dataset('filename', (total_subjects,), dtype=dt)
        # f.swmr_mode = True

        os.makedirs(data_dir + '/examples/', exist_ok=True)

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

            image_to_save = np.reshape(normalise_zero_one(t1_data), (1,) + (target_size))

            plt.imshow(image_to_save[0, target_size[0] // 2 + 5, :, :, 0])
            plt.savefig(data_dir + '/examples/' + data_point['qc_label'] + '_' + data_point['candidate_id'] + '.png')
            plt.axis('off')
            plt.tight_layout()
            plt.clf()

            image_to_save = adjust_log(image_to_save)

            f['MRI'][i, ...] = image_to_save
            f['filename'][i] = data_point['t1_filename']

            plt.imshow(image_to_save[0, target_size[0] // 2 + 5, :, :, 0])
            plt.savefig(data_dir + '/examples/' + data_point['qc_label'] + '_' + data_point['candidate_id'] + '_log_normalized.png')
            plt.axis('off')
            plt.tight_layout()
            plt.clf()

            print(str(i+1), 'of', total_subjects, data_point['candidate_id'])

def ibis_bids(source_dir, label_file):

    ibis_dir = '/data1/users/adoyle/IBIS/'
    # sample_nifti_file = 'E:/brains/MRBrainS18/training/1/orig/FLAIR.nii.gz'
    #
    # sample_nii = nib.load(sample_nifti_file)
    #
    # sample_header = sample_nii.get_header()

    ibis_files = []

    for participant_level in os.listdir(ibis_dir):
        if os.path.isdir(ibis_dir + participant_level):
            for session_level in os.listdir(ibis_dir + participant_level):
                if os.path.exists(ibis_dir + participant_level + '/' + session_level + '/mri/native/') and not 'sub' in session_level:
                    # print(session_level)
                    for filename in os.listdir(ibis_dir + participant_level + '/' + session_level + '/mri/native/'):
                        if '.mnc' in filename and not 'phantom' in filename.lower():
                            minc_filepath = ibis_dir + participant_level + '/' + session_level + '/mri/native/' + filename
                            ibis_files.append(minc_filepath)
                            # print(minc_filepath)

    for ibis_img in ibis_files:

        tokens = ibis_img.split('_')
        subj_id = tokens[1]
        session = tokens[2]
        run = tokens[4][:-4]

        full_path = ibis_dir + '/BIDS/sub-' + subj_id + '/ses-' + session.upper() + '/anat/'

        os.makedirs(full_path, exist_ok=True)

        new_filename = 'sub-' + subj_id + '_ses-' + session.upper() + '_run-' + run + '_T1w'

        subprocess.run('mnc2nii -nii ' + ibis_img + ' ' + full_path + new_filename, shell=True, check=True)


def redo_extension():
    ibis_dir = '/data1/users/adoyle/IBIS/BIDS/'

    for subject_level in os.listdir(ibis_dir):
        if 'sub' in subject_level:
            for session_level in os.listdir(ibis_dir + subject_level):
                for t1_filename in os.listdir(ibis_dir + subject_level + '/' + session_level + '/anat/'):
                    if '.nii.gz.nii' in t1_filename:
                        os.rename(ibis_dir + subject_level + '/' + session_level + '/anat/' + t1_filename, ibis_dir + subject_level + '/' + session_level + '/anat/' + t1_filename[:-7])


if __name__ == '__main__':
    print('Creating IBIS HDF5 file for quality control training')

    # source_dir = 'E:/brains/IBIS/'
    # label_file = 'ibis_t1_qc.csv'

    make_ibis_qc()

    # ibis_bids(source_dir, label_file)
    # redo_extension()
    print('Done!')