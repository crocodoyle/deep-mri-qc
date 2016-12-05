from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Convolution3D, MaxPooling3D, Flatten, BatchNormalization, SpatialDropout3D
from keras.optimizers import SGD

import numpy as np
import h5py

import os
import nibabel
import cPickle as pkl

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

    model.add(Convolution3D(7, 3, 3, 3, activation='relu', input_shape=(1, 160, 256, 224)))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
#    model.add(SpatialDropout2D(0.5))

    model.add(Convolution3D(8, 3, 3, 3, activation='relu'))
    model.add(BatchNormalization())
#    model.add(MaxPooling2D(pool_size=(3, 3)))
#    model.add(SpatialDropout2D(0.5))

    model.add(Convolution3D(12, 3, 3, 3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
#    model.add(SpatialDropout2D(0.2))
#
    model.add(Flatten())
    model.add(Dense(10, init='uniform', activation='relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(10, init='uniform', activation='relu'))
    model.add(Dense(nb_classes, init='uniform'))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=["accuracy"])

    return model

# generator that produces batches of size n so that we don't overload memory
def batch(train_indices, labels, n):
    f = h5py.File('ibis.hdf5', 'r')
    images = f.get('ibis_t1')

    x_train = np.zeros((n, 1, 160, 256, 224), dtype=np.float32)
    y_train = np.zeros((n, 2), dtype=np.int8)

    indices = train_indices
    while True:
        np.random.shuffle(indices)

        samples_this_batch = 0
        for i, index in enumerate(indices):
            x_train[i%n, 0, :, :, :] = images[index]
            y_train[i%n, :]   = labels[index]
            samples_this_batch += 1
            if (i+1) % n == 0:
                yield (x_train, y_train)
                samples_this_batch = 0
            elif i == len(indices)-1:
                yield (x_train[0:samples_this_batch, ...], y_train[0:samples_this_batch, :])
		samples_this_batch = 0



if __name__ == "__main__":
    print "Running automatic QC"
    fail_data = "/home/adoyle/T1_Minc_Fail"
    pass_data = "/home/adoyle/T1_Minc_Pass"

    train_indices, test_indices, labels, filename_test = load_data(fail_data, pass_data)

    # define model
    model = qc_model()

    # print summary of model
    model.summary()

    num_epochs = 100

    # for epoch in range(num_epochs):
	   # print 'epoch', epoch, 'of', str(num_epochs)
    model.fit_generator(batch(train_indices, labels, 2), nb_epoch=num_epochs, samples_per_epoch=len(train_indices), validation_data=batch(test_indices, labels, 2), nb_val_samples=len(test_indices))

    model_config = model.get_config()
    pkl.dumps(model_config, 'convnet_model' + str(num_epochs) + '.pkl')


    score = model.evaluate_generator(batch(test_indices, labels, 2), len(test_indices))
    print score
