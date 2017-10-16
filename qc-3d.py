from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv3D, MaxPooling3D, Flatten, BatchNormalization
from keras.callbacks import ModelCheckpoint

import numpy as np
import h5py

import keras.backend as K

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


workdir = '/data1/data/deepqc/'


def qc_model():
    nb_classes = 3

    conv_size = (5, 5, 5)
    pool_size = (2, 2, 2)

    model = Sequential()

    model.add(Conv3D(8, conv_size, activation='relu', input_shape=(192, 256, 256, 1)))
    # model.add(Dropout(0.2))
    model.add(Conv3D(8, conv_size, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=pool_size))

    model.add(Conv3D(16, conv_size, activation='relu'))
    # model.add(Dropout(0.2))
    model.add(Conv3D(16, conv_size, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=pool_size))


    model.add(Conv3D(32, conv_size, activation='relu'))
    # model.add(Dropout(0.3))
    model.add(Conv3D(32, conv_size, activation='relu'))
    # model.add(Dropout(0.3))
    # model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=pool_size))
#
    model.add(Conv3D(64, conv_size, activation='relu'))
    # model.add(Dropout(0.4))

    # model.add(Conv3D(256, (1, 1, 1), activation=('relu')))
    # model.add(Dropout(0.5))
    model.add(Conv3D(nb_classes, (1, 1, 1), activation=('relu')))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=["accuracy", sens_spec])

    return model

def batch(indices, f):
    images = f['MRI']
    labels = f['qc_label']    #already in one-hot

    while True:
        np.random.shuffle(indices)

        for index in indices:
            try:
                # print(images[index, ...][np.newaxis, ...].shape)
                yield (np.reshape(images[index, ...], (192, 256, 256, 1))[np.newaxis, ...], labels[index, ...][np.newaxis, ...])
            except:
                yield (np.reshape(images[index, ...], (192, 256, 256, 1))[np.newaxis, ...])

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

# sensitivity = true positives / (true positives + false negatives)
# specificity = true negatives / (true negatives + false positives)
def sens_spec(y_true, y_pred):
    true_positives, true_negatives, false_positives, false_negatives = 0, 0, 0, 0

    total = y_true.shape[0]
    print('samples', total)

    for index in range(total):
        y = y_true[index, ...]
        y_hat = y_pred[index, ...]

        y_int = K.argmax(y)
        y_hat_int = K.argmax(y_hat)

        if y_int >= 1:
            if y_int == y_hat_int:
                true_positives += 1
            else:
                false_positives += 1
        else:
            if y_int == y_hat_int:
                true_negatives += 1
            else:
                false_negatives += 1

    return (true_positives / (true_positives + false_negatives + 0.0001), true_negatives / (true_negatives + false_negatives + 0.0001))



if __name__ == "__main__":

    ping_end_index = 1153
    abide_end_index = 2255
    ibis_end_index = 2723
    ds030_end_index = 2988

    f = h5py.File(workdir + 'deepqc.hdf5')

    ping_indices = list(range(0, ping_end_index))
    abide_indices = list(range(ping_end_index, abide_end_index))
    ibis_indices = list(range(abide_end_index, ibis_end_index))
    ds030_indices = list(range(ibis_end_index, ds030_end_index))

    # print('ping:', ping_indices)
    # print('abide:', abide_indices)
    # print('ibis:', ibis_indices)
    # print('ds030', ds030_indices)

    train_indices = ping_indices + abide_indices + ibis_indices

    print('PING samples:', len(ping_indices))
    print('ABIDE samples:', len(abide_indices))
    print('IBIS samples:', len(ibis_indices))
    print('training samples:', len(train_indices), len(ping_indices) + len(abide_indices) + len(ibis_indices))

    train_labels = np.zeros((len(train_indices), 3))
    print('labels shape:', train_labels.shape)

    for index in train_indices:
        label = f['qc_label'][index, ...]
        train_labels[index, ...] = label

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
        scores = model.test_on_batch(f['MRI'][index, ...], f['qc_label'][index, ...])
        print(scores)

    plot_training_error(hist)