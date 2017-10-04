from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv3D, MaxPooling3D, Flatten, BatchNormalization

import numpy as np
import h5py

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

def qc_model():
    nb_classes = 3

    model = Sequential()

    model.add(Conv3D(8, (3, 3, 3), activation='relu', input_shape=(1, 192, 256, 256)))
    model.add(Dropout(0.2))
    model.add(Conv3D(8, (3, 3, 3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))

    model.add(Conv3D(16, (3, 3, 3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv3D(16, (3, 3, 3), activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))


    model.add(Conv3D(32, (3, 3, 3), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Conv3D(32, (3, 3, 3), activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
#
    model.add(Conv3D(64, (3, 3, 3), activation='relu'))
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(10, init='uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, init='uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, init='uniform'))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=["accuracy"])

    return model

def batch(indices, f):
    images = f['MRI']
    labels = f['qc_label']    #already in one-hot

    while True:
        np.random.shuffle(indices)

        for index in indices:
            try:
                yield (images[index, ...], labels[index, ...])
            except:
                yield (images[index, ...])

if __name__ == "__main__":

    ping_end_index = 1153
    abide_end_index = 2124
    ibis_end_index = 2592
    ds030_end_index = 2857

    f = h5py.File('/data1/data/deepqc/deepqc.hdf5')

    ping_indices = np.linspace(0, ping_end_index)
    abide_indices = np.linspace(ping_end_index + 1, abide_end_index)
    ibis_indices = np.linspace(abide_end_index + 1, ibis_end_index)
    ds030_indices = np.linspace(ibis_end_index + 1, ds030_end_index)

    train_indices = ping_indices + abide_indices + ibis_indices

    train_labels = f['qc_label'][train_indices]

    skf = StratifiedKFold(n_splits=1, test_size = 0.1)

    train_indices, validation_indices = skf.split(train_indices, train_labels)

    test_indices = ds030_indices

    print('train:', train_indices)
    print('test:', test_indices)


    # define model
    model = qc_model()

    # print summary of model
    model.summary()

    num_epochs = 300

    hist = model.fit_generator(
        batch(train_indices, f),
        len(train_indices),
        epochs=num_epochs,
        samples_per_epoch=len(train_indices),
        validation_data=batch(validation_indices, f),
        validation_steps=len(validation_indices)
    )
