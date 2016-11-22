from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Convolution3D, MaxPooling3D, Flatten, BatchNormalization, SpatialDropout3D
from keras.optimizers import SGD

import numpy as np
import h5py

import os
import nibabel

import matplotlib.pyplot as plt

from sklearn.cross_validation import StratifiedShuffleSplit

def load_data(fail_path, pass_path):
    print "loading data..."
    filenames = []
    labels = []

    f = h5py.File('ibis.hdf5', 'w')


    # First loop through the data we need to count the number of files
    # also check dims
    numImgs = 0
    x_dim, y_dim, z_dim = 0, 0, 0
    for root, dirs, files in os.walk(fail_path, topdown=False):
	for name in files:
            numImgs += 1
	    if x_dim == 0:
               img =  nibabel.load(os.path.join(root, name)).get_data()
               print np.shape(img)
               x_dim = np.shape(img)[0]
               y_dim = np.shape(img)[1]
               z_dim = np.shape(img)[2]
    for root, dirs, files in os.walk(pass_path, topdown=False):
        for name in files:
            numImgs += 1

    images = f.create_dataset('ibis_t1', (numImgs, x_dim, y_dim, z_dim), dtype='float32')
    labels = np.zeros((numImgs, 2), dtype='bool')

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

    indices = StratifiedShuffleSplit(labels, test_size=0.4, n_iter=1, random_state=None)

    train_index, test_index = None, None
    for train_indices, test_indices in indices:
        train_index = train_indices
        test_index  = test_indices

    filename_test = []
    for i, f in enumerate(filenames):
        if i in test_index:
            filename_test.append(f)

    return train_index, test_index, labels, filename_test


def qc_model():
#    data_dim = 160*256*224
    nb_classes = 2

    model = Sequential()

    model.add(Convolution3D(16, 12, 15, 15, border_mode='same', input_shape=(1, 160, 256, 224)))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(4, 4, 4)))
    model.add(BatchNormalization())
#    model.add(SpatialDropout2D(0.5))

    model.add(Convolution3D(12, 12, 12, 12, border_mode='same'))
    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(3, 3)))
#    model.add(SpatialDropout2D(0.5))

    model.add(Convolution3D(12, 5, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
#    model.add(SpatialDropout2D(0.2))
#
    model.add(Convolution3D(12, 3, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    #    model.add(SpatialDropout2D(0.5))

    model.add(Convolution3D(12, 3, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(SpatialDropout3D(0.4))

    model.add(Convolution3D(12, 2, 2, 2, border_mode='same'))
    model.add(Activation('relu'))
    model.add(SpatialDropout3D(0.5))

    model.add(Flatten())
    model.add(Dense(256, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, init='uniform'))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=["accuracy"])

    return model

# def model_train(x_train, x_test, y_train, y_test, filename_test):

#     print "shape of training data:", np.shape(x_train)
#     print "shape of testing data:", np.shape(x_test)
#     print "shape of training labels:", np.shape(y_train)
#     print "shape of testing labels:", np.shape(y_test)
#     print "filename list:", len(filename_test)



#     model.fit_generator(train_batch, nb_epoch=1, batch_size=50)
#     #should return model to workspace so that I can keep training it

#     score = model.evaluate(x_test, y_test, batch_size=10)
#     print model.metrics_names
#     print score


if __name__ == "__main__":
    print "Running automatic QC"
    fail_data = "/home/adoyle/T1_Minc_Fail"
    pass_data = "/home/adoyle/T1_Minc_Pass"

    train_indices, test_indices, labels, filename_test = load_data(fail_data, pass_data)

    model = qc_model()

    num_epochs = 100

    for epoch in range(num_epochs):
        print "training epoch..."
        model.fit_generator(train_batch(train_indices, labels, 10), nb_epoch=1)



    # score = model.evaluate(images)

    #chooo chooooo

# generator that produces training batches of size n so that we don't overload memory
def train_batch(train_indices, labels, n):
    f = h5py.File('ibis.hdf5', 'r')
    images = f.get('ibis_t1')

    x_train = np.zeros((n, 1, 160, 256, 224))
    y_train = np.zeros((n, 2))

    while True:
        indices = shuffle(train_indices)

        samples_this_batch = 0
        for i, index in enumerate(indices):
            x_train[i, ...] = images[index]
            y_train[i, :]   = labels[index]
            samples_this_batch += 1
            if i % n == 0:
                yield (x_train, y_train)
                samples_this_batch = 0
            elif i == len(indices)-1:
                yield (x_train[0:samples_this_batch, ...], y_train[0:samples_this_batch, :])

