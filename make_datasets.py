import numpy as np
from scipy.spatial.distance import euclidean

import os, sys, time, csv

import h5py
import sklearn

from sklearn.neighbors import KDTree

import nibabel as nib


output_path = '/data1/data/ABIDE/'


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


    f = h5py.File(output_path + 'abide.hdf5', 'w')
    f.create_dataset('images', (2295, 361, 433, 361, 3), dtype='float32') # t1, gradient magnitude, surface distance
    f.create_dataset('surfaces', (2295, 81920*2, 3))
    f.create_dataset('filenames', (2295,), dtype=h5py.special_dtype(vlen=unicode))
    f.create_dataset('labels', (2295,), dtype='bool')

    patient_key = {}

    i = 0
    for filename in os.listdir(path + '/surfaces/'):
        if 'white_surface' in filename and not 'followup' in filename and 'anat_1' in filename:
            patient_id = filename.split('+')[1]
            patient_key[str(patient_id)] = i

            surface_obj = open(os.path.join(path + '/surfaces/', filename), 'r')
            surface_obj.readlines(1)
            j = 0
            for line in surface_obj.readlines():
                coords = line.split(" ")
                if len(coords) != 3:
                    break
                f['surfaces'][i, j, :] = [float(coords[0]) + 72.25, float(coords[1]) + 126.25, float(coords[2] + 90.25)]
                j += 1
            surface_obj.close()


            i += 1

        else:
            continue

    for filename in os.listdir(path + '/T1s/'):

        if not 'followup' in filename and 'anat_1' in filename:
            patient_id = filename.split('+')[1]
            i = patient_key[str(patient_id)]

            img = nib.load(os.path.join(path + '/T1s/', filename)).get_data()
            print('image shape:', np.shape(img))

            f['images'][i,:,:,:,0] = img


            grad = np.gradient(img)
            print('gradient shape:', np.shape(grad))

            f['images'][i,:,:,:,1] = np.sum(grad, axis=0)


            surface_distance = np.ones((361, 433, 361), dtype='float32')


            surf_points = f['surfaces'][i,:,:]


            print("surface points: ", np.shape(surf_points))

            floatX = np.zeros(np.shape(surface_distance)[0])
            floatY = np.zeros(np.shape(surface_distance)[1])
            floatZ = np.zeros(np.shape(surface_distance)[2])


            print("building KDTree...")
            tree = KDTree(surf_points, leaf_size=1000)
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


    label_file = open(os.path.join(path, label_file))
    for line in label_file.readlines():

        patient_id = line[0].split('+')[1].split('_')[0]
        label = int(line[1])

        i = patient_key[patient_id]
        f['labels'][i] = label

    print('all done!')

    f.close()

    return 0




if __name__ == "__main__":
    make_abide('/data1/data/ABIDE/', 'labels.csv')
  # make_nihpd('/data1/data/NIHPD/assembly/', 'data1/data/dl-datasets/')
