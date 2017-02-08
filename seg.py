from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, BatchNormalization, SpatialDropout2D, merge
from keras.layers import Convolution3D, MaxPooling3D, SpatialDropout3D, UpSampling3D

from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils.visualize_util import plot

from keras import backend as K

import numpy as np
import h5py

import os
import nibabel

import cPickle as pkl

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import confusion_matrix

import argparse


images_dir = '/data1/MICCAI_2012_MALF'
scratch_dir = images_dir

def load_oasis():
    print "loading data..."
    patient_id = []

    f = h5py.File(scratch_dir + 'oasis.hdf5', 'w')

    # First loop through the data we need to count the number of files
    # also check dims
    numImgs = 0
    x_dim, y_dim, z_dim = 0, 0, 0
    for root, dirs, files in os.walk(images_dir, topdown=False):
        for name in files:
            if "T1_norm-stx152lin.nii.gz" in name:
                print root
                numImgs += 1
                if x_dim == 0:
                   img =  nibabel.load(os.path.join(root, name)).get_data()

                   print np.shape(img)
                   x_dim = np.shape(img)[0]
                   y_dim = np.shape(img)[1]
                   z_dim = np.shape(img)[2]

    print "There are", numImgs, "images" #should be 35
    images = f.create_dataset('oasis', (numImgs, x_dim, y_dim, z_dim), dtype='float32')
    labels = f.create_dataset('oasis_labels', (numImgs, x_dim, y_dim, z_dim), dtype='short')

    i = 0
    for root, dirs, files in os.walk(images_dir, topdown=False):
        for name in files:
            if "T1_norm-stx152lin.nii.gz" in name:
                img = nibabel.load(os.path.join(root, name)).get_data()
                if "T1_norm-stx152lin.nii.gz" in name and np.shape(img) == (x_dim, y_dim, z_dim):
                    images[i] = img
                if "labels.nii.gz" in name and np.shape(img) == (x_dim, y_dim, z_dim):
                    labels[i] = img

                    patient_id.append(root)
                    i += 1

    f.close()

    index = range(numImgs)

    ss = ShuffleSplit(n_splits=1, test_size=0.2)

    train_index, other_index    = ss.split(index).next()
    validation_index            = other_index[::2]  # even
    test_index                  = other_index[1::2] # odd

    print "training images:", len(train_index)
    print "validation images:", len(validation_index)
    print "test_index:", len(test_index)

    return train_index, validation_index, test_index, patient_id

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def segmentation_model():
    nb_classes = 4

    inputs = Input(shape=(1, 181, 217, 181))

    conv1 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2,3,2))(conv1)

    conv2 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2,2,2))(conv2)

    conv3 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2,2,2))(conv3)

    conv4 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2,2,2))(conv4)

    conv5 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(conv5)
    pool5 = MaxPooling3D(pool_size=(2,2,2))(conv5)

    conv6 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(pool5)
    conv6 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(conv6)
    pool6 = MaxPooling3D(pool_size=(2,2,2))(conv6)

    conv7 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(pool6)
    conv7 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling3D(size=(2,2,2))(conv7), conv6], mode='concat', concat_axis=1)
    conv8 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling3D(size=(2,2,2))(conv7), conv4], mode='concat', concat_axis=1)
    conv9 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(conv9)

    up10 = merge([UpSampling3D(size=(2,2,2))(conv8), conv3], mode='concat', concat_axis=1)
    conv10 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(up10)
    conv10 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(conv10)

    up11 = merge([UpSampling3D(size=(2,2,2))(conv9), conv2], mode='concat', concat_axis=1)
    conv11 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(up11)
    conv11 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(conv11)

    conv12 = Convolution3D(13, 1, 1, 1, activation='sigmoid')(conv11)

    model = Model(input=inputs, output=conv12)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def batch(indices, labels, n, random_slice=False):
    f = h5py.File(scratch_dir + 'oasis.hdf5', 'r')
    images = f.get('oasis')
    labels = f.get('oasis_labels')

    x_train = np.zeros((n, 1, 181, 217, 181), dtype=np.float32)
    y_train = np.zeros((n, 181, 217, 181), dtype=np.int8)

    while True:
        np.random.shuffle(indices)

        samples_this_batch = 0
        for i, index in enumerate(indices):
            x_train[i%n, :, :, :] = images[index, :, :, :]
            y_train[i%n, :, :, :] = labels[index, :, :, :]
            samples_this_batch += 1
            if (i+1) % n == 0:
                yield (x_train, y_train)
                samples_this_batch = 0
            elif i == len(indices)-1:
                yield (x_train[0:samples_this_batch, ...], y_train[0:samples_this_batch, :])
        samples_this_batch = 0

def test_images(model):
    f = h5py.File(scratch_dir + 'oasis.hdf5', 'r')

    images = f.get('oasis')
    labels = f.get('oasis_labels')

    model.predict()


    dice = 0


    return dice

if __name__ == "__main__":
    print "Running segmentation"

    train_indices, validation_indices, test_indices, patient_id = load_oasis()

    model = segmentation_model()
    model.summary()

    plot(model, to_file="segmentation_model.png")


    model_checkpoint = ModelCheckpoint("models/best_segmentation_model.hdf5", monitor="val_acc", verbose=0, save_best_only=True, save_weights_only=False, mode='auto')

    hist = model.fit_generator(batch(train_indices, labels, 2,True), nb_epoch=400, samples_per_epoch=len(train_indices), validation_data=batch(validation_indices, labels, 2), nb_val_samples=len(validation_indices), callbacks=[model_checkpoint], class_weight = {0:.7, 1:.3})


    model.load_weights('models/best_segmentation_model.hdf5')
