from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Input
from keras.callbacks import ModelCheckpoint

from keras.layers.merge import add, concatenate

from dltk.core.io.augmentation import flip, elastic_transform

from keras.optimizers import SGD

import numpy as np
import h5py
import pickle

import keras.backend as K

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from custom_loss import sensitivity, specificity

import tensorflow as tf


# These 4 lines suposedly enable distributed GPU training
# server = tf.train.Server.create_local_server()
# sess = tf.Session(server.target)
#
# from keras import backend as K
# K.set_session(sess)


workdir = '/home/users/adoyle/'

image_size = (192, 256, 192)
slice_size = (192, 256)


def top_model():
    nb_classes = 2
    conv_size = (5, 5)
    pool_size = (2, 2)

    inputs = [Input(shape=(192, 256, 192)), Input(shape=(192, 192, 256)), Input(shape=(192, 256, 192))]

    # XY plane
    xy_conv1 = Conv2D(16, conv_size, activation='relu')(inputs[0])
    xy_norm1 = BatchNormalization()(xy_conv1)
    xy_drop1 = Dropout(0.2)(xy_norm1)
    xy_pool1 = MaxPooling2D(pool_size=pool_size)(xy_drop1)

    xy_conv2 = Conv2D(32, conv_size, activation='relu')(xy_pool1)
    xy_norm2 = BatchNormalization()(xy_conv2)
    xy_drop2 = Dropout(0.2)(xy_norm2)
    xy_pool2 = MaxPooling2D(pool_size=pool_size)(xy_drop2)

    xy_conv3 = Conv2D(64, conv_size, activation='relu')(xy_pool2)
    xy_norm3 = BatchNormalization()(xy_conv3)
    xy_drop3 = Dropout(0.2)(xy_norm3)
    xy_pool3 = MaxPooling2D(pool_size=pool_size)(xy_drop3)

    xy_conv4 = Conv2D(128, conv_size, activation='relu')(xy_pool3)
    xy_norm4 = BatchNormalization()(xy_conv4)
    xy_drop4 = Dropout(0.2)(xy_norm4)
    xy_pool4 = MaxPooling2D(pool_size=pool_size)(xy_drop4)

    xy_fully = Conv2D(10, (1, 1), activation='relu')(xy_pool4)
    xy_flat  = Flatten()(xy_fully)

    # XZ plane
    xz_conv1 = Conv2D(16, conv_size, activation='relu')(inputs[1])
    xz_norm1 = BatchNormalization()(xz_conv1)
    xz_drop1 = Dropout(0.2)(xz_norm1)
    xz_pool1 = MaxPooling2D(pool_size=pool_size)(xz_drop1)

    xz_conv2 = Conv2D(32, conv_size, activation='relu')(xz_pool1)
    xz_norm2 = BatchNormalization()(xz_conv2)
    xz_drop2 = Dropout(0.2)(xz_norm2)
    xz_pool2 = MaxPooling2D(pool_size=pool_size)(xz_drop2)

    xz_conv3 = Conv2D(64, conv_size, activation='relu')(xz_pool2)
    xz_norm3 = BatchNormalization()(xz_conv3)
    xz_drop3 = Dropout(0.2)(xz_norm3)
    xz_pool3 = MaxPooling2D(pool_size=pool_size)(xz_drop3)

    xz_conv4 = Conv2D(128, conv_size, activation='relu')(xz_pool3)
    xz_norm4 = BatchNormalization()(xz_conv4)
    xz_drop4 = Dropout(0.2)(xz_norm4)
    xz_pool4 = MaxPooling2D(pool_size=pool_size)(xz_drop4)

    xz_fully = Conv2D(10, (1, 1), activation='relu')(xz_pool4)
    xz_flat = Flatten()(xz_fully)

    # YZ plane
    yz_conv1 = Conv2D(16, conv_size, activation='relu')(inputs[2])
    yz_norm1 = BatchNormalization()(yz_conv1)
    yz_drop1 = Dropout(0.2)(yz_norm1)
    yz_pool1 = MaxPooling2D(pool_size=pool_size)(yz_drop1)

    yz_conv2 = Conv2D(32, conv_size, activation='relu')(yz_pool1)
    yz_norm2 = BatchNormalization()(yz_conv2)
    yz_drop2 = Dropout(0.2)(yz_norm2)
    yz_pool2 = MaxPooling2D(pool_size=pool_size)(yz_drop2)

    yz_conv3 = Conv2D(64, conv_size, activation='relu')(yz_pool2)
    yz_norm3 = BatchNormalization()(yz_conv3)
    yz_drop3 = Dropout(0.2)(yz_norm3)
    yz_pool3 = MaxPooling2D(pool_size=pool_size)(yz_drop3)

    yz_conv4 = Conv2D(128, conv_size, activation='relu')(yz_pool3)
    yz_norm4 = BatchNormalization()(yz_conv4)
    yz_drop4 = Dropout(0.2)(yz_norm4)
    yz_pool4 = MaxPooling2D(pool_size=pool_size)(yz_drop4)

    yz_fully = Conv2D(10, (1, 1), activation='relu')(yz_pool4)
    yz_flat = Flatten()(yz_fully)

    allplanes = concatenate([xy_flat, xz_flat, yz_flat])
    all_drop = Dropout(0.5)(allplanes)

    output = Dense(nb_classes, activation='softmax')(all_drop)

    model = Model(inputs=inputs, outputs=[output])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=["accuracy", sensitivity, specificity])

    return model

def top_batch(indices, f):
    images = f['MRI']
    labels = f['qc_label']    #already in one-hot

    while True:
        np.random.shuffle(indices)

        for index in indices:
            try:
                t1_image = f['MRI'][index, ...]

                t1_image = flip(t1_image, 2)
                # t1_image = elastic_transform(t1_image, [3,3,3], [3,3,3])

                xy = t1_image[np.newaxis, ...]
                xz = np.swapaxes(t1_image, 1, 2)[np.newaxis, ...]
                yz = np.swapaxes(t1_image, 0, 2)[np.newaxis, ...]

                yield ([xy, xz, yz], labels[index, ...][np.newaxis, ...])
            except:
                yield ([xy, xz, yz])


def plot_metrics(hist):
    epoch_num = range(len(hist.history['acc']))
    # train_error = np.subtract(1, np.array(hist.history['acc']))
    # test_error  = np.subtract(1, np.array(hist.history['val_acc']))

    plt.clf()
    plt.plot(epoch_num, np.array(hist.history['acc']), label='Training Accuracy')
    plt.plot(epoch_num, np.array(hist.history['val_acc']), label="Validation Accuracy")
    # plt.plot(epoch_num, np.array(hist.history['sensitivity']), label="Training Sensitivity")
    # plt.plot(epoch_num, np.array(hist.history['specificity']), label="Validation Accuracy")

    plt.legend(shadow=True)
    plt.xlabel("Training Epoch Number")
    plt.ylabel("Accuracy")
    plt.savefig(workdir + 'training-results.png')
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
    model = top_model()

    # print summary of model
    model.summary()

    num_epochs = 100

    model_checkpoint = ModelCheckpoint( workdir + 'best_qc_model.hdf5',
                                        monitor="val_acc",
                                        save_best_only=True)

    hist = model.fit_generator(
        top_batch(train_indices, f),
        len(train_indices),
        epochs=num_epochs,
        callbacks=[model_checkpoint],
        validation_data=top_batch(validation_indices, f),
        validation_steps=len(validation_indices),
        use_multiprocessing=True
    )

    model.load_weights(workdir + 'best_qc_model.hdf5')
    model.save(workdir + 'qc_model.hdf5')

    scores = model.predict_generator(top_batch(test_indices, f), len(test_indices))
    print(scores)

    plot_metrics(hist)