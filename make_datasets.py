import numpy as np
from scipy.spatial.distance import euclidean

import os, sys, time, csv, subprocess

import h5py
import sklearn

from sklearn.neighbors import KDTree

import nibabel as nib

from multiprocessing import Pool


output_path = '/data1/data/ABIDE/'
cores = 12


def make_ibis(input_path, output_path, label_file):
    f = h5py.File(output_path + 'ibis.hdf5', 'w')

    #store a maximum number of "pass" images
    max_pass = 1000

    # First loop through the data we need to count the number of files
    # also check dims
    numImgs = 0
    x_dim, y_dim, z_dim = 0, 0, 0
    for root, dirs, files in os.walk(fail_path, topdown=False):
        for name in files:
            numImgs += 1
            if x_dim == 0:
               img =  nibabel.load(os.path.join(root, name)).get_data()
               print(np.shape(img))
               x_dim = np.shape(img)[0]
               y_dim = np.shape(img)[1]
               z_dim = np.shape(img)[2]
    for root, dirs, files in os.walk(pass_path, topdown=False):
        for name in files:
            numImgs += 1
            if numImgs > max_pass:
                break
        if numImgs > max_pass:
            break

    images    = f.create_dataset('T1', (numImgs, x_dim, y_dim, z_dim), dtype='float32')
    labels    = f.create_dataset('T1_QC_labels', (numImgs,2), dtype='bool')
    filenames = f.create_dataset('T1_filenames', (numImgs,), dtype='str')

    # Second time through, write the image data to the HDF5 file
    i = 0
    for root, dirs, files in os.walk(fail_path, topdown=False):
        for name in files:
            img = nibabel.load(os.path.join(root, name)).get_data()
            if np.shape(img) == (x_dim, y_dim, z_dim):
                images[i] = img
                labels[i] = [1, 0]
                filenames.append(os.path.join(root, name))
                i += 1


    for root, dirs, files in os.walk(pass_path, topdown=False):
        for name in files:
            img = nibabel.load(os.path.join(root, name)).get_data()
            if np.shape(img) == (x_dim, y_dim, z_dim):
                images[i] = img
                labels[i] = [0, 1]
                filenames.append(os.path.join(root, name))
                i += 1
            if i > max_pass:
                break
        if i > max_pass:
            break

    return

def make_nihpd(input_path, output_path, label_file):

  f = h5py.File(output_path + 'nihpd.hdf5', 'w')



def train_test_val(val_frac, test_frac, labels):
    sss_validation = StratifiedShuffleSplit(n_splits=1, test_size=val_frac + test_frac, random_state=42)
    sss_test       = StratifiedShuffleSplit(n_splits=1, test_size=test_frac / (val_frac+test_frac), random_state=42)

    train_indices, validation_indices, test_indices = None, None, None

    for train_index, validation_index in sss_validation.split(np.zeros(len(labels)), labels):
        train_indices      = train_index
        validation_indices = validation_index

    for validation_index, test_index in sss_test.split(np.zeros(len(labels[validation_indices])), labels[validation_indices]):
        validation_indices = validation_index
        test_indices       = test_index
    print("training images:", len(train_index))
    print("validation images:", len(validation_index))
    print("test_index:", len(test_index))

    return train_indices, validation_indices, test_indices


def make_abide(path, label_file):


    patient_data = {}

    print("Reading QC labels...")
    label_file = open(os.path.join(path, label_file))
    lines = label_file.readlines()


    #establish a unique id as id number - scan-number - session-number
    for i, line in enumerate(lines[1:]):   # skip header
        patient_id = line.split('+')[1].split('_')[0]
        label = int(line.split(',')[1])


        anat_pos = line.find('anat')
        if anat_pos > 0:
            anat = line[anat_pos + 5]
        else:
            anat = "0"

        followup_pos = line.find('followup')
        if followup_pos > 0:
            followup = line[followup_pos + 9]
        else:
            followup = "0"


        patient_id += '-' + anat + '-' + followup

        patient_data[patient_id] = {}
        patient_data[patient_id]['label'] = label
        patient_data[patient_id]['index'] = i

    total_subjects = i
    print(total_subjects, "in QC file")

    f = h5py.File(output_path + 'abide.hdf5', 'w')
    f.create_dataset('images', (total_subjects+100, 361, 433, 361, 3), dtype='float32') # t1, gradient magnitude, surface distance
    f.create_dataset('surfacepoints', (total_subjects+100, 40962*2, 3))
    f.create_dataset('filenames', (total_subjects+100,), dtype=h5py.special_dtype(vlen=unicode))
    f.create_dataset('labels', (total_subjects+100,), dtype='bool')


    #load labels
    num = 0
    for patient_id in patient_data:
        # print(patient_id)
        i = int(patient_data[patient_id]['index'])
        l = int(patient_data[patient_id]['label'])

        f['labels'][i] = l

        num += 1

    print(num, 'patients in labels')


    #load surface points
    for filename in os.listdir(path + '/surfaces/'):

        patient_id = filename.split('+')[1]

        anat_pos = filename.find('anat')
        if anat_pos > 0:
            anat = filename[anat_pos + 5]
        else:
            anat = "0"

        followup_pos = filename.find('followup')
        if followup_pos > 0:
            followup = filename[followup_pos + 9]
        else:
            followup = "0"

        patient_id += '-' + anat + '-' + followup


        i = patient_data[patient_id]['index']


        if "right" in filename:
            patient_data[patient_id]['surfacefile-right'] = os.path.join(path + '/surfaces/', filename)
            surface_obj = open(patient_data[patient_id]['surfacefile-right'])
            j = 0
        elif "left" in filename:
            patient_data[patient_id]['surfacefile-left'] = os.path.join(path + '/surfaces/', filename)
            surface_obj = open(patient_data[patient_id]['surfacefile-left'])
            j = 40962


        surface_obj.readlines(1)

        for line in surface_obj.readlines():
            coords = line.split(" ")
            if len(coords) != 3:
                break
            f['surfacespoints'][i, j, :] = [float(coords[0]) + 72.25, float(coords[1]) + 126.25, float(coords[2]) + 90.25]
            j += 1
        surface_obj.close()

    #load T1, compute gradient image
    # for index, filename in enumerate(os.listdir(path + '/T1s/')):

    #     patient_id = filename.split('+')[1]

    #     anat_pos = filename.find('anat')
    #     if anat_pos > 0:
    #         anat = filename[anat_pos + 5]
    #     else:
    #         anat = "0"

    #     followup_pos = filename.find('followup')
    #     if followup_pos > 0:
    #         followup = filename[followup_pos + 9]
    #     else:
    #         followup = "0"

    #     patient_id += '-' + anat + '-' + followup
    #     i = patient_data[patient_id]['index']

    #     # print(i)
    #     img = nib.load(os.path.join(path + '/T1s/', filename)).get_data()

    #     # print('image shape:', np.shape(img))

    #     f['images'][i,:,:,:,0] = img


    #     grad = np.gradient(img)
    #     # print('gradient shape:', np.shape(grad))

    #     f['images'][i,:,:,:,1] = np.sum(grad, axis=0)

    #     print(index, 'of', len(os.listdir(path + '/T1s/')))

    print("Computing surface distances... Could take a while")



    surf_points = np.zeros((40962*2, 3), dtype='float32')

    p = Pool(cores)

    for i in range(total_subjects):
        surf_points = f['surfacepoints'][i,:,:]
        p.map(distance_to_surf, args=(surf_points, i,), callback = save_result)

        print("Done ", str(i))

    p.close()
    p.join()


    f.close()



    return 0

def save_result(vol_info):

    nib.save(vol_info['surface'], os.path.join(output_path, str(vol_info['index']) + '.nii.gz'))


def distance_to_surf(surface_points, index):
    surface_distance = np.ones((361, 433, 361), dtype='float32')

    print("surface points: ", np.shape(surf_points))

    floatX = np.zeros(np.shape(surface_distance)[0])
    floatY = np.zeros(np.shape(surface_distance)[1])
    floatZ = np.zeros(np.shape(surface_distance)[2])


    print("building KDTree...")
    tree = KDTree(surf_points, leaf_size=10000)
    print("built KDTree!")

    for z in range(np.shape(surface_distance)[0]):
        print("z: ", z)
        for y in range(np.shape(surface_distance)[1]):
            for x in range(np.shape(surface_distance)[2]):
                (distance, index) = tree.query(np.reshape([floatZ[z], floatY[y], floatX[x]], (1,3)), return_distance = True)
                surface_distance[z,y,x] = distance
                # brute force method, doesn't work very well
                # for point in surf_points:
                #     d = euclidean([floatZ[z], floatY[y], floatX[x]], point)

                #     if surface_distance[z,y,x] > d:
                #         surface_distance[z,y,x] = d


    print('done ', filename)
    result = {}
    result['index'] = index
    result['surface'] = surface_distance
    return result


def combine_objs(obj1, obj2, newname):
    print(obj1)
    print(obj2)
    print(newname)
    subprocess.Popen(['objconcat', obj1, obj2, 'none', 'none', newname, 'none'])



if __name__ == "__main__":
    path = '/data1/data/ABIDE/'

    for filename in os.listdir(path + '/surfaces/'):
        if "right" in filename:
            # patient_id = filename.split('+')[1]

            filename1 = filename
            filename2 = filename.replace("right", "left")
            filename3 = filename.replace("right", "combined")

            combine_objs(os.path.join(path, filename1), os.path.join(path, filename2), os.path.join(path, filename3))


    # make_abide('/data1/data/ABIDE/', 'labels.csv')
  # make_nihpd('/data1/data/NIHPD/assembly/', 'data1/data/dl-datasets/')
