from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv3D, MaxPooling3D, Flatten, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam

import numpy as np
import h5py
import pickle

import keras.backend as K
import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from custom_loss import sensitivity, specificity

workdir = '/home/users/adoyle/deepqc/'
data_file = 'deepqc-all-sets.hdf5'

image_size = (192, 256, 192)


def qc_model():
    nb_classes = 2

    conv_size = (3, 3, 3)
    pool_size = (2, 2, 2)

    model = Sequential()

    model.add(Conv3D(4, conv_size, activation='relu', input_shape=(image_size[0], image_size[1], image_size[2], 1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Conv3D(8, conv_size, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=pool_size))

    model.add(Conv3D(16, conv_size, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv3D(16, conv_size, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())
    # model.add(MaxPooling3D(pool_size=pool_size))

    # model.add(Conv3D(64, conv_size, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Conv3D(32, conv_size, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(BatchNormalization())
    # model.add(MaxPooling3D(pool_size=pool_size))
#
    model.add(Conv3D(32, conv_size, activation='relu'))
    # model.add(Dropout(0.4))
    # model.add(MaxPooling3D(pool_size=pool_size))

    # model.add(Conv3D(64, conv_size, activation='relu'))
    # # model.add(Dropout(0.4))
    # model.add(MaxPooling3D(pool_size=pool_size))

    model.add(Conv3D(32, conv_size, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Conv3D(32, conv_size, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Conv3D(8, (1, 1, 1), activation='relu'))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))

    # model.add(Conv3D(256, (1, 1, 1), activation=('relu')))
    # model.add(Dropout(0.5))
    # model.add(Conv3D(nb_classes, (1, 1, 1), activation=('relu')))
    # model.add(Dropout(0.5))
    # model.add(Flatten())
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
                yield (np.reshape(images[index, ...], image_size + (1,))[np.newaxis, ...], labels[index, ...][np.newaxis, ...])
            except:
                yield (np.reshape(images[index, ...], image_size + (1,))[np.newaxis, ...])

def plot_metrics(hist, results_dir):
    epoch_num = range(len(hist.history['acc']))
    train_error = np.subtract(1, np.array(hist.history['acc']))
    test_error  = np.subtract(1, np.array(hist.history['val_acc']))

    plt.clf()
    plt.plot(epoch_num, np.array(hist.history['acc']), label='Training Accuracy')
    plt.plot(epoch_num, np.array(hist.history['val_acc']), label="Validation Error")
    plt.legend(shadow=True)
    plt.xlabel("Training Epoch Number")
    plt.ylabel("Error")
    plt.savefig(results_dir + 'results.png')
    plt.close()

def setup_experiment(workdir):
    try:
        experiment_number = pickle.load(open(workdir + 'experiment_number.pkl', 'rb'))
        experiment_number += 1
    except:
        print('Couldnt find the file to load experiment number')
        experiment_number = 0

    print('This is experiment number:', experiment_number)

    results_dir = workdir + '/experiment-' + str(experiment_number) + '/'
    os.makedirs(results_dir)

    pickle.dump(experiment_number, open(workdir + 'experiment_number.pkl', 'wb'))

    return results_dir, experiment_number

if __name__ == "__main__":
    results_dir, experiment_number = setup_experiment(workdir)

    abide_indices = pickle.load(open(workdir + 'abide_indices.pkl', 'rb'))
    ds030_indices = pickle.load(open(workdir + 'ds030_indices.pkl', 'rb'))
    ibis_indices = pickle.load(open(workdir + 'ibis_indices.pkl', 'rb'))
    ping_indices = pickle.load(open(workdir + 'ping_indices.pkl', 'rb'))

    f = h5py.File(workdir + data_file, 'r')
    images = f['MRI']

    print('number of samples in dataset:', images.shape[0])

    # print('ping:', ping_indices)
    # print('abide:', abide_indices)
    # print('ibis:', ibis_indices)
    # print('ds030', ds030_indices)

    train_indices = ping_indices + abide_indices + ibis_indices
    # train_indices = abide_indices

    # print('PING samples:', len(ping_indices))
    # print('ABIDE samples:', len(abide_indices))
    # print('IBIS samples:', len(ibis_indices))
    # print('training samples:', len(train_indices), len(ping_indices) + len(abide_indices) + len(ibis_indices))

    train_labels = np.zeros((len(train_indices), 2))
    print('labels shape:', train_labels.shape)

    good_subject_index = 0
    for index in train_indices:
        label = f['qc_label'][index, ...]
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

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)

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
        validation_steps=len(validation_indices)
    )

    model.load_weights(results_dir + 'best_qc_model.hdf5')
    model.save(results_dir + 'qc_model.hdf5')



    metrics = model.evaluate_generator(batch(test_indices, f), len(test_indices))

    print(model.metrics_names)
    print(metrics)

    pickle.dump(metrics, open(results_dir + 'test_metrics', 'wb'))

    plot_metrics(hist, results_dir)

    print('This experiment brought to you by the number:', experiment_number)