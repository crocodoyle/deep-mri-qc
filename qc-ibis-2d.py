from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, BatchNormalization, SpatialDropout2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils.visualize_util import plot

import numpy as np
import h5py

import os
import nibabel

import pickle as pkl

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix

import argparse


images_dir = '/gs/scratch/adoyle/'
cluster = False

if cluster:
    images_dir  = '/gs/scratch/adoyle/'
    scratch_dir = os.environ.get('RAMDISK') + '/'
else:
    images_dir   = '/data1/data/IBIS/'
    scratch_dir  = images_dir

print('SCRATCH', scratch_dir)
print('IMAGES:', images_dir)


def load_data(fail_path, pass_path):
    print("loading data...")
    filenames = []
    labels = []

    f = h5py.File(scratch_dir + 'ibis.hdf5', 'w')


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
            if i > max_pass:
                break
        if i > max_pass:
            break

    sss_validation = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    sss_test       = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

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

    return train_index, validation_index, test_index, labels, filenames

def load_in_memory(train_index, test_index, labels):
    f = h5py.File(scratch_dir + 'ibis.hdf5', 'r')
    images = f.get('ibis_t1')

    x_train = np.array(images)[train_index]
    y_train = np.array(labels)[train_index]
    x_test  = np.array(images)[test_index]
    y_test  = np.array(labels)[test_index]

    return x_train, x_text, y_train, y_test

def qc_model():
    nb_classes = 2

    model = Sequential()

    model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=(256, 224, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.2))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SpatialDropout2D(0.3))

    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SpatialDropout2D(0.3))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(SpatialDropout2D(0.3))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(SpatialDropout2D(0.3))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SpatialDropout2D(0.4))

    model.add(Convolution2D(256, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.5))

    model.add(Flatten())
    model.add(Dense(256, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())

    model.add(Dense(nb_classes, init='uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=1e-3, momentum=0.9, decay=1e-6, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=["accuracy"])

    return model

def batch(indices, labels, n, random_slice=False):
    f = h5py.File(scratch_dir + 'ibis.hdf5', 'r')
    images = f.get('ibis_t1')

    x_train = np.zeros((n, 256, 224, 1), dtype=np.float32)
    y_train = np.zeros((n, 2), dtype=np.int8)

    while True:
        np.random.shuffle(indices)

        samples_this_batch = 0
        for i, index in enumerate(indices):
            if random_slice:
                rn=np.random.randint(-4,4)
            else:
                rn=0
            x_train[i%n, :, :, 0] = images[index, 80+rn, :, :]
            y_train[i%n, :]   = labels[index]
            samples_this_batch += 1
            if (i+1) % n == 0:
                yield (x_train, y_train)
                samples_this_batch = 0
            elif i == len(indices)-1:
                yield (x_train[0:samples_this_batch, ...], y_train[0:samples_this_batch, :])
        samples_this_batch = 0

def test_images(model, test_indices, labels, filename_test, slice_modifier, save_imgs=False):
    f = h5py.File(scratch_dir + 'ibis.hdf5', 'r')
    images = f.get('ibis_t1')

    predictions = np.zeros((len(test_indices)))
    actual = np.zeros((len(test_indices)))

    predict_batch = np.zeros((1, 256, 224, 1))

    print("test indices:", len(test_indices))
    print("test index max:", max(test_indices))
    print("labels:", len(labels))
    print("filenames:", len(filename_test))

    for i, index in enumerate(test_indices):
        predict_batch[0,:,:,0] = images[index,80+slice_modifier, :,:]

        prediction = model.predict_on_batch(predict_batch)[0][0]
        if prediction >= 0.5:
            predictions[i] = 1
        else:
            predictions[i] = 0
        actual[i] = labels[index][0]

        if save_imgs:
            plt.imshow(images[index,80,:,:])
            if predictions[i] == 1 and actual[i] == 1:
                plt.savefig('/home/adoyle/images/fail_right_' + os.path.basename(filename_test[i]) + ".png")
            elif predictions[i] == 0 and actual[i] == 0:
                plt.savefig('/home/adoyle/images/pass_right_' + os.path.basename(filename_test[i]) + '.png')
            elif predictions[i] == 1 and actual[i] == 0:
                plt.savefig('/home/adoyle/images/pass_wrong_' + os.path.basename(filename_test[i]) + '.png')
            elif predictions[i]  == 0 and actual[i] == 1:
                plt.savefig('/home/adoyle/images/fail_wrong_' + os.path.basename(filename_test[i]) + '.png')
            plt.clf()

    conf = confusion_matrix(actual, predictions)
    print('Confusion Matrix')
    print(conf)

    print(np.shape(conf))

    tp = conf[0][0]
    tn = conf[1][1]
    fp = conf[0][1]
    fn = conf[1][0]

    print('true negatives:', tn)
    print('true positives:', tp)
    print('false negatives:', fn)
    print('false positives:', fp)

    sensitivity = float(tp) / (float(tp) + float(fn))
    specificity = float(tn) / (float(tn) + float(fp))


    print('sens:', sensitivity)
    print('spec:', specificity)

    return sensitivity, specificity

if __name__ == "__main__":
    print("Running automatic QC")
    fail_data = images_dir + "T1_Minc_Fail"
    pass_data = images_dir + "T1_Minc_Pass"

    parser = argparse.ArgumentParser()
    parser.add_argument("--imagesdir", help="directory that contains the input images")
    parser.add_argument("--cluster", help="specifies whether training is done on a cluster")
    parser.add_argument("--scratchdir", help="directory to use for gathering image data on cluster")
    parser.add_argument("--train", help="specifies to train a new model")
    parser.add_argument("--model", help="path to model file to load, to continue a training run or just do testing")
    parser.add_argument("--epochs", help="number of epochs to train the model", type=int)
    args = parser.parse_args()

    if args.imagesdir is not None:
        images_dir = args.imagesdir
    if args.cluster is not None:
        cluster = args.cluster
    if args.scratchdir is not None:
        scratch_dir = args.scratchdir
    if args.train is not None:
        do_training = True
    if args.model is not None:
        load_model = True
        model_to_load = args.model
    if args.epochs is not None:
        nb_epoch = args.epochs

    print("command line arguments")
    print(args)

    train_indices, validation_indices, test_indices, labels, filenames = load_data(fail_data, pass_data)

    model = qc_model()
    model.summary()
    plot(model, to_file="model.png")

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.001)
    stop_early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    model_checkpoint = ModelCheckpoint(images_dir + "models/best_model.hdf5", monitor="val_acc", verbose=0, save_best_only=True, save_weights_only=False, mode='auto')

    hist = model.fit_generator(batch(train_indices, labels, 2,True), nb_epoch=400, samples_per_epoch=len(train_indices), validation_data=batch(validation_indices, labels, 2), nb_val_samples=len(validation_indices), callbacks=[model_checkpoint], class_weight = {0:.7, 1:.3})


    model.load_weights(images_dir + 'models/best_model.hdf5')


    test_scores = []
    sensitivities = []
    specificities = []

    for test_run in range(-5, 5):
        score = model.evaluate_generator(batch(test_indices, labels, 2, True), len(test_indices))
        test_scores.append(score[1])


        if test_run == 0:
            sens, spec = test_images(model, test_indices, labels, filenames, test_run, save_imgs=True)
        else:
            sens, spec = test_images(model, test_indices, labels, filenames, test_run, save_imgs=False)
        print("sensitivity:", sens)
        print("specificity:", spec)

        sensitivities.append(sens)
        specificities.append(spec)

    print('scores:', test_scores)
    print('average score', np.mean(test_scores))
    print('average sensitivity', np.mean(sensitivities))
    print('average specificity', np.mean(specificities))


    print(model.metrics_names)

    print(hist.history.keys())

    epoch_num = range(len(hist.history['acc']))
    train_error = np.subtract(1,np.array(hist.history['acc']))
    test_error  = np.subtract(1,np.array(hist.history['val_acc']))

    plt.clf()
    plt.plot(epoch_num, train_error, label='Training Error')
    plt.plot(epoch_num, test_error, label="Validation Error")
    plt.legend(shadow=True)
    plt.xlabel("Training Epoch Number")
    plt.ylabel("Error")
    plt.savefig('results.png')
    plt.close()
