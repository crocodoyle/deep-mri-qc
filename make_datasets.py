import numpy as np
from scipy.spatial.distance import euclidean

import os, sys, time, csv, subprocess

from dltk.core.io.preprocessing import normalise_zero_one, resize_image_with_crop_or_pad

import h5py
from skimage.transform import resize
from sklearn.neighbors import KDTree

import nibabel as nib

from multiprocessing import Pool, Process

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

output_dir = '/data1/data/deepqc/'
output_file = '/data1/data/deepqc/deepqc.hdf5'
cores = 10

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

exemplar_file = '/data1/data/PING/p0007_20100128_144417_2_mri.mnc'

def make_ping(input_path, f, label_file, subject_index):
    with open(os.path.join(input_path, label_file)) as label_file:
        qc_reader = csv.reader(label_file)

        for line in qc_reader:
            try:
                t1_filename = line[0][0:-4] + '.mnc'
                label = int(line[1])                                                #0, 1, or 2
                comment = line[2]

                t1_data = nib.load(input_path + t1_filename).get_data()

                if not t1_data.shape == (192, 256, 256):
                    print('resizing from', t1_data.shape)
                    t1_data = resize_image_with_crop_or_pad(t1_data, img_size=[192, 256, 256], mode='constant')

                f['MRI'][subject_index, ...] = normalise_zero_one(t1_data)

                if label == 0:
                    f['qc_label'][subject_index, :] = [1, 0, 0]
                elif label == 1:
                    f['qc_label'][subject_index, :] = [0, 1, 0]
                elif label == 2:
                    f['qc_label'][subject_index, :] = [0, 0, 1]

                f['qc_comment'][subject_index] = comment

                print(subject_index, t1_filename, np.shape(t1_data))

                plt.imshow(t1_data[96, ...])
                plt.axis('off')
                plt.savefig(output_dir + t1_filename[:-4] + '.png', bbox_inches='tight')

                subject_index += 1
            except FileNotFoundError as e:
                print('File not found:', line)

    return subject_index


def make_ibis(input_path, f, label_file, subject_index):
    with open(os.path.join(input_path, label_file)) as label_file:
        qc_reader = csv.reader(label_file)

        for line in qc_reader:
            try:
                t1_filename = line[0][0:-4] + '.mnc'
                label = int(line[1])                                                #0, 1, or 2

                t1_data = nib.load(input_path + t1_filename).get_data()

                if not t1_data.shape == (192, 256, 256):
                    print('resizing from', t1_data.shape)
                    t1_data = resize_image_with_crop_or_pad(t1_data, img_size=[192, 256, 256], mode='constant')

                f['MRI'][subject_index, ...] = normalise_zero_one(t1_data)

                if label == 0:
                    f['qc_label'][subject_index, :] = [1, 0, 0]
                elif label == 1:
                    f['qc_label'][subject_index, :] = [0, 1, 0]
                elif label == 2:
                    f['qc_label'][subject_index, :] = [0, 0, 1]

                print(subject_index, t1_filename)

                plt.imshow(t1_data[96, ...])
                plt.axis('off')
                plt.savefig(output_dir + t1_filename[:-4] + '.png', bbox_inches='tight')

                subject_index += 1
            except FileNotFoundError as e:
                print('File not found:', line)

    return subject_index

def make_abide(input_path, f, label_file, subject_index):
    with open(os.path.join(input_path, label_file)) as label_file:
        qc_reader = csv.reader(label_file)
        qc_reader.__next__()

        for line in qc_reader:
            try:
                t1_filename = line[0] + '.mnc'

                resample_command = ['mincresample',
                                    '-clobber',
                                    '-nearest',
                                    '-unsigned',
                                    '-byte',
                                    '-keep_real_range',
                                    '-like',
                                    exemplar_file,
                                    input_path + t1_filename,
                                    input_path + "/resampled/" + t1_filename]

                one_hot = [0, 0, 0]

                total_labels = 0
                if len(line[2]) > 0:
                    label1 = int(line[2]) + 1                #-1, 0, or 1
                    one_hot[label1] = 1
                    total_labels += 1
                if len(line[3]) > 0:
                    label2 = int(line[3]) + 1
                    one_hot[label2] = 1
                    total_labels += 1
                if len(line[4]) > 0:
                    label3 = int(line[4]) + 1
                    one_hot[label3] = 1
                    total_labels += 1

                one_hot = np.multiply(one_hot, 1/total_labels)

                f['qc_label'][subject_index, :] = one_hot

                subprocess.run(resample_command)

                t1_data = nib.load(input_path + '/resampled/' + t1_filename).get_data()

                if not t1_data.shape == (192, 256, 256):
                    print('resizing from', t1_data.shape)
                    # if t1_data.shape[1] > 400:
                    #     print('resampling from', t1_data.shape)
                    #     t1_data = resize(t1_data, (t1_data.shape[0]/2, t1_data.shape[1]/2, t1_data.shape[2]/2), order=1)

                    t1_data = resize_image_with_crop_or_pad(t1_data, img_size=[192, 256, 256], mode='constant')


                f['MRI'][subject_index, ...] = normalise_zero_one(t1_data)

                print(subject_index, t1_filename)

                plt.imshow(t1_data[96, ...])
                plt.axis('off')
                plt.savefig(output_dir + t1_filename[:-4] + '.png', bbox_inches='tight')

                subject_index += 1
            except FileNotFoundError as e:
                print('File not found:', line)

    return subject_index

def make_ds030(input_path, f, label_file, subject_index):
    with open(os.path.join(input_path, label_file)) as label_file:
        qc_reader = csv.reader(label_file)
        qc_reader.__next__()

        for line in qc_reader:
            try:
                t1_filename = line[0] + '.nii.gz'
                label = line[8]

                if len(label) > 0:

                    resample_command = ['mincresample',
                                        '-clobber',
                                        '-nearest',
                                        '-unsigned',
                                        '-byte',
                                        '-keep_real_range',
                                        '-like',
                                        exemplar_file,
                                        output_file]

                    subprocess.run(['nii2mnc', input_path + t1_filename])
                    subprocess.run(resample_command)

                    t1_filename = t1_filename[:-7] + '.mnc'
                    t1_data = nib.load(input_path + t1_filename).get_data()

                    if not t1_data.shape == (192, 256, 256):
                        print('resizing from', t1_data.shape)
                        t1_data = resize_image_with_crop_or_pad(t1_data, img_size=[192, 256, 256], mode='constant')

                    f['MRI'][subject_index, ...] = normalise_zero_one(t1_data)

                    if 'ok' in label:
                        one_hot = [0, 0, 1]
                    elif 'maybe' in label:
                        one_hot = [0, 1, 0]
                    elif 'exclude' in label:
                        one_hot = [1, 0, 0]

                    f['qc_label'][subject_index, :] = one_hot

                    print(subject_index, t1_filename)

                    plt.imshow(t1_data[96, ...])
                    plt.axis('off')
                    plt.savefig(output_dir + t1_filename[:-4] + '.png', bbox_inches='tight')

                    subject_index += 1

            except FileNotFoundError as e:
                print('File not found:', line)

    return subject_index

def make_abide_surfaces(path, label_file):
    patient_data = {}

    for index, filename in enumerate(os.listdir(path + '/T1_downsampled/')):
        if '.mnc' in filename:
            patient_id = filename[:-4]
            patient_data[patient_id] = {}

            print(filename, patient_id)
            patient_data[patient_id]['index'] = index
        else:
            index -= 1

    total_subjects = index + 1
    print('total of ', total_subjects, 'downsampled T1s')

    f = h5py.File(output_path + 'abide.hdf5', 'w')
    # f.create_dataset('images', (total_subjects, 181, 217, 181, 3), dtype='float32')  # t1, gradient magnitude, surface distance
    f.create_dataset('surfacepoints', (total_subjects, 40962*2, 3))
    # f.create_dataset('filenames', (total_subjects,), dtype=h5py.special_dtype(vlen=unicode))
    f.create_dataset('labels', (total_subjects,), dtype='bool')


    # load images and compute gradient
    # for filename in os.listdir(path + '/T1_downsampled/'):
    #     if '.mnc' in filename:
    #         patient_id = filename.split('.')[0]
    #
    #         i = patient_data[patient_id]['index']
    #         print(i, 'of', len(os.listdir(path + '/T1_downsampled/')))
    #
    #         img = nib.load(os.path.join(path + '/T1_downsampled/', filename)).get_data()  # load image data
    #
    #         f['images'][i, :, :, :, 0] = img
    #         f['images'][i, :, :, :, 1] = np.sum(np.gradient(img), axis=0)
    #     else:
    #         print(filename, 'should not be here')

    # extract surfaces from combined left/right surface objs
    for filename in os.listdir(path + '/surfaces/'):
        patient_id = filename.split('+')[1]

        i = patient_data[patient_id]['index']
        surface_obj = open(path + '/surfaces/' + filename)
        surface_obj.readlines(1) # first line is a header

        print('patient', i)

        for j, line in enumerate(surface_obj.readlines()):
            coords = line.split(" ")
            print('xyz coordinates', coords)
            if len(coords) != 3:
                break
            f['surfacespoints'][i, j, :] = [float(coords[0]) + 72.5, float(coords[1]) + 126.5,
                                            float(coords[2]) + 90.5]

        surface_obj.close()

    print("Reading QC labels...")
    label_file = open(os.path.join(path, label_file))
    lines = label_file.readlines()

    for i, line in enumerate(lines[1:]):   # skip header
        patient_id = line.split('+')[1].split('_')[0]
        label = int(line.split(',')[1])

        print('patient:', patient_id, 'label:', label, 'line: ', line)

        if not 'followup' in line and not 'anat_2' in line and not 'anat_3' in line and not 'anat_4' in line and not 'anat_5' in line and not 'anat_6' in line and not 'baseline' in line:
            patient_data[patient_id]['label'] = label

    print("Computing surface distances... Could take a while")

    # compute surface distance volumes in same space as T1
    p = Pool(cores)
    surf_points = np.zeros((40962*2, 3), dtype='float32')

    for i in range(total_subjects):
        surf_points = f['surfacepoints'][i, :, :]

        # p.apply_async(distance_to_surf, args=(surf_points, i,))
        distance_to_surf(surf_points, i)
        print("Launched job", i)

    p.close()
    p.join()

    print("Done ", total_subjects, 'surfaces')

    f.close()

    return 0

def distance_to_surf(surface_points, patient_id):
    surface_distance = np.ones((181, 217, 181), dtype='float32')

    print("surface points: ", np.shape(surface_points))

    floatX = np.zeros(np.shape(surface_distance)[0], dtype='float32')
    floatY = np.zeros(np.shape(surface_distance)[1], dtype='float32')
    floatZ = np.zeros(np.shape(surface_distance)[2], dtype='float32')

    for xx in range(np.shape(floatX)[0]):
        floatX[xx] = float(xx)

    for yy in range(np.shape(floatY)[0]):
        floatY[yy] = float(yy)

    for zz in range(np.shape(floatZ)[0]):
        floatZ[zz] = float(zz)

    print("building KDTree...")
    tree = KDTree(surface_points, leaf_size=10000)
    print("built KDTree!")

    for z in range(np.shape(surface_distance)[0]):
        print("z: ", z)
        for y in range(np.shape(surface_distance)[1]):
            for x in range(np.shape(surface_distance)[2]):
                (distance, index) = tree.query(np.reshape([floatZ[z], floatY[y], floatX[x]], (1, 3)), return_distance = True)
                surface_distance[z, y, x] = distance
                # brute force method, very slow
                # for point in surf_points:
                #     d = euclidean([floatZ[z], floatY[y], floatX[x]], point)

                #     if surface_distance[z,y,x] > d:
                #         surface_distance[z,y,x] = d

    output_filename = os.path.join(output_path, str(patient_id) + '_surface_distance.nii.gz')
    img = nib.Nifti1Image(surface_distance, np.eye(4))

    nib.save(img, output_filename)
    return surface_distance, output_filename


def combine_objs(obj1, obj2, newname):
    print(obj1)
    print(obj2)
    print(newname)
    subprocess.Popen(['objconcat', obj1, obj2, 'none', 'none', newname, 'none'])

if __name__ == "__main__":
    path = '/data1/data/ABIDE/'

    # for filename in os.listdir(path + '/surfaces/'):
    #     if "right" in filename:
    #         # patient_id = filename.split('+')[1]
    #
    #         filename1 = filename
    #         filename2 = filename.replace("right", "left")
    #         filename3 = filename.replace("right", "combined")
    #
    #         combine_objs(os.path.join(path, 'surfaces/' + filename1), os.path.join(path, 'surfaces/' + filename2), os.path.join(path, 'surfaces/' + filename3))

    # for filename in os.listdir(path + '/T1s/'):
    #     try:
    #         patient_id = filename.split('+')[1]
    #
    #         p = subprocess.Popen(['mincresample', '-nearest_neighbour', '-like', path + 'icbm_template_1.00mm.mnc', path + 'T1s/' + filename, path + 'T1_downsampled/' + patient_id + '.mnc', '-clobber'])
    #         p.communicate()
    #     except:
    #         print filename



    #PING: 1154
    #IBIS: 468
    #ABIDE: 1113
    #ds030: 282

    total_subjects = 1154 + 468 + 1113 + 282

    f = h5py.File(output_file, 'w')
    # f.create_dataset('MRI', (1154+468+113+282, 192, 256, 256), maxshape=(None, 192, 256, 256), dtype='float32')
    f.create_dataset('MRI', (total_subjects, 192, 256, 256), dtype='float32')
    f.create_dataset('qc_label', (total_subjects, 3), maxshape=(None, 3), dtype='uint8')
    dt = h5py.special_dtype(vlen=bytes)
    f.create_dataset('qc_comment', (total_subjects,), dtype=dt)

    subject_index = 0

    ping_end_index, abide_end_index, ibis_end_index, ds030_end_index = 0, 0, 0, 0
    # ping_end_index = make_ping('/data1/data/PING/', f, 't1_qc.csv', subject_index) - 1
    abide_end_index = make_abide('/data1/data/deep_abide/', f, 't1_qc.csv', ping_end_index + 1) - 1
    ibis_end_index = make_ibis('/data1/data/IBIS/', f, 'ibis_t1_qc.csv', abide_end_index + 1) - 1
    ds030_end_index = make_ds030('/data1/data/ds030/', f, 'ds030_DB.csv', ibis_end_index + 1) - 1

    print(ping_end_index, abide_end_index, ibis_end_index, ds030_end_index)
    print(1154+468+1113+282)

    f.close()

    # make_abide('/data1/data/ABIDE/', 'labels.csv')
  # make_nihpd('/data1/data/NIHPD/assembly/', 'data1/data/dl-datasets/')
