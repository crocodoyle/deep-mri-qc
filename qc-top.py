from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.callbacks import ModelCheckpoint

import numpy as np
import h5py
import pickle

import keras.backend as K

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from custom_loss import sensitivity, specificity

workdir = '/data1/data/deepqc/'

image_size = (192, 256, 192)
slice_size = (192, 256)


def qc_model():
    nb_classes = 2

    conv_size = (3, 3)
    pool_size = (2, 2)

    model = Sequential()

    model.add(Conv2D(64, conv_size, activation='relu', input_shape=(192, 256, 192)))
    model.add(BatchNormalization())
    model.add(Conv2D(64, conv_size, activation='relu'))

    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Conv2D(64, conv_size, activation='relu'))
    model.add(Conv2D(64, conv_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Conv2D(64, conv_size, activation='relu'))
    model.add(Conv2D(64, conv_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Conv2D(64, conv_size, activation='relu'))
    model.add(Conv2D(64, conv_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Conv2D(64, conv_size, activation='relu'))
    model.add(Conv2D(64, conv_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=["accuracy", sensitivity, specificity])

    return model

def batch(indices, f):
    images = f['MRI']
    labels = f['qc_label']    #already in one-hot

    while True:
        np.random.shuffle(indices)

        for index in indices:
            try:
                # print(images[index, ...][np.newaxis, ...].shape)

                label = [labels[index, 0], labels[index, 1] + labels[index, 2]][np.newaxis, ...]
                print('label shape', label.shape)

                yield tuple(images[index, ...][np.newaxis, ...], label)
            except:
                yield tuple(images[index, ...][np.newaxis, ...])

def plot_training_error(hist):
    epoch_num = range(len(hist.history['acc']))
    train_error = np.subtract(1, np.array(hist.history['acc']))
    test_error  = np.subtract(1, np.array(hist.history['val_acc']))

    plt.clf()
    plt.plot(epoch_num, train_error, label='Training Error')
    plt.plot(epoch_num, test_error, label="Validation Error")
    plt.legend(shadow=True)
    plt.xlabel("Training Epoch Number")
    plt.ylabel("Error")
    plt.savefig(workdir + 'results.png')
    plt.close()

if __name__ == "__main__":

    abide_indices = pickle.load(open(workdir + 'abide_indices.pkl', 'rb'))
    ds030_indices = pickle.load(open(workdir + 'ds030_indices.pkl', 'rb'))

    f = h5py.File(workdir + 'deepqc.hdf5', 'r')

    # ping_indices = list(range(0, ping_end_index))
    # abide_indices = list(range(ping_end_index, abide_end_index))
    # ibis_indices = list(range(abide_end_index, ibis_end_index))
    # ds030_indices = list(range(ibis_end_index, ds030_end_index))

    # print('ping:', ping_indices)
    # print('abide:', abide_indices)
    # print('ibis:', ibis_indices)
    # print('ds030', ds030_indices)

    # train_indices = ping_indices + abide_indices + ibis_indices
    train_indices = abide_indices

    # print('PING samples:', len(ping_indices))
    # print('ABIDE samples:', len(abide_indices))
    # print('IBIS samples:', len(ibis_indices))
    # print('training samples:', len(train_indices), len(ping_indices) + len(abide_indices) + len(ibis_indices))


    train_labels = np.zeros((len(abide_indices), 2))
    print('labels shape:', train_labels.shape)

    good_subject_index = 0
    for index in train_indices:
        label = [f['qc_label'][index, 0], f['qc_label'][index, 1] + f['qc_label'][index, 2]]
        print(label)
        train_labels[good_subject_index, ...] = label
        good_subject_index += 1

    skf = StratifiedShuffleSplit(n_splits=1, test_size = 0.1)

    for train, val in skf.split(train_indices, train_labels):
        train_indices = train
        validation_indices = val

    test_indices = ds030_indices

    print('train:', train_indices)
    print('test:', test_indices)


    # define model
    model = qc_model()

    # print summary of model
    model.summary()

    num_epochs = 300

    model_checkpoint = ModelCheckpoint( workdir + 'best_qc_model.hdf5',
                                        monitor="val_acc",
                                        save_best_only=True)

    hist = model.fit_generator(
        batch(train_indices, f),
        len(train_indices),
        epochs=num_epochs,
        callbacks=[model_checkpoint],
        validation_data=batch(validation_indices, f),
        validation_steps=len(validation_indices),
        use_multiprocessing=True
    )

    model.load_weights(workdir + 'best_qc_model.hdf5')
    model.save(workdir + 'qc_model.hdf5')

    predicted = []
    actual = []

    for index in test_indices:
        scores = model.test_on_batch(f['MRI'][index, ...][np.newaxis, ...], [f['qc_label'][index, 0], f['qc_label'][index, 1] + f['qc_label'][index, 2]][np.newaxis, ...])
        print(scores)

    plot_training_error(hist)