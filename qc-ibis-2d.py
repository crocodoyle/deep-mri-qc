from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

import numpy as np
import h5py

import os, csv
import nibabel as nib

from dltk.core.io.preprocessing import normalise_zero_one, resize_image_with_crop_or_pad
from custom_loss import sensitivity, specificity


import pickle as pkl

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix

import argparse

workdir = '/home/users/adoyle/deepqc/IBIS'
datadir = '/data1/data/IBIS/'

label_file = datadir + 'ibis_t1_qc.csv'

total_subjects = 468
target_size = (168, 256, 244)

def make_ibis_qc():
    f = h5py.File(workdir + 'ibis.hdf5', 'w')
    f.create_dataset('ibis_t1', (total_subjects, target_size[0], target_size[1], target_size[2]), dtype='float32')
    f.create_dataset('qc_label', (total_subjects, 2), dtype='float32')

    index = 0

    indices = []
    labels = []

    with open(label_file, 'r') as labels_csv:
        qc_reader = csv.reader(labels_csv)

        for line in qc_reader:
            try:
                t1_filename = line[0][0:-4] + '.mnc'
                label = int(line[1])  # 0, 1, or 2

                if label >= 1:
                    one_hot = [0.0, 1.0]
                else:
                    one_hot = [1.0, 0.0]

                f['qc_label'][index, :] = one_hot
                t1_data = nib.load(datadir + t1_filename).get_data()

                if not t1_data.shape == target_size:
                    # print('resizing from', t1_data.shape)
                    t1_data = resize_image_with_crop_or_pad(t1_data, img_size=target_size, mode='constant')

                f['MRI'][index, ...] = normalise_zero_one(t1_data)

                # plt.imshow(t1_data[96, ...])
                # plt.axis('off')
                # plt.savefig(output_dir + t1_filename[:-4] + '.png', bbox_inches='tight', cmap='gray')

                indices.append(index)
                labels.append(np.argmax(one_hot))

                index += 1
            except Exception as e:
                print('Error:', e)

    f.close()

    return indices, labels


def qc_model():
    nb_classes = 2

    conv_size = (3, 3)
    pool_size = (2, 2)

    model = Sequential()

    model.add(Conv2D(16, conv_size, activation='relu', input_shape=(target_size[1], target_size[2], 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, conv_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(32, conv_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, conv_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(64, conv_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, conv_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, conv_size, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(nb_classes, activation='softmax'))

    sgd = SGD(lr=1e-3, momentum=0.9, decay=1e-6, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=["accuracy", sensitivity, specificity])

    return model

def batch(indices, n, random_slice=False):
    f = h5py.File(workdir + 'ibis.hdf5', 'r')
    images = f['ibis_t1']
    labels = f['qc_label']

    x_train = np.zeros((n, target_size[1], target_size[2], 1), dtype=np.float32)
    y_train = np.zeros((n, 2), dtype=np.int8)

    while True:
        np.random.shuffle(indices)

        samples_this_batch = 0
        for i, index in enumerate(indices):
            if random_slice:
                rn=np.random.randint(-4, 4)
            else:
                rn=0
            x_train[i%n, :, :, 0] = images[index, target_size[0]+rn, :, :]
            y_train[i%n, :]   = labels[index]
            samples_this_batch += 1
            if (i+1) % n == 0:
                yield (x_train, y_train)
                samples_this_batch = 0
            elif i == len(indices)-1:
                yield (x_train[0:samples_this_batch, ...], y_train[0:samples_this_batch, :])


def test_images(model, test_indices, labels, filename_test, slice_modifier, save_imgs=False):
    f = h5py.File(workdir + 'ibis.hdf5', 'r')
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

    try:
        experiment_number = pkl.load(open(workdir + 'experiment_number.pkl', 'rb'))
        experiment_number += 1
    except:
        print('Couldnt find the file to load experiment number')
        experiment_number = 0

    print('This is experiment number:', experiment_number)

    results_dir = workdir + '/experiment-' + str(experiment_number) + '/'
    os.makedirs(results_dir)

    pkl.dump(experiment_number, open(workdir + 'experiment_number.pkl', 'wb'))

    indices, labels = make_ibis_qc()

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3)
    train_indices, test_indices = sss.split(indices, labels)

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5)
    test_indices, validation_indices = sss2.split(test_indices, labels[test_indices])

    model = qc_model()
    model.summary()

    model_checkpoint = ModelCheckpoint(results_dir + "best_model.hdf5", monitor="val_acc", verbose=0, save_best_only=True, save_weights_only=False, mode='auto')

    hist = model.fit_generator(batch(train_indices, 32, True), len(train_indices), epochs=400, validation_data=batch(validation_indices, 32), validation_steps=len(validation_indices), callbacks=[model_checkpoint], class_weight = {0:.7, 1:.3})

    model.load_weights(results_dir + 'best_model.hdf5')

    # test_scores = []
    # sensitivities = []
    # specificities = []
    #
    # for test_run in range(-5, 5):
    #     score = model.evaluate_generator(batch(test_indices, 2, True), len(test_indices))
    #     test_scores.append(score[1])
    #
    #
    #     if test_run == 0:
    #         sens, spec = test_images(model, test_indices, labels, filenames, test_run, save_imgs=True)
    #     else:
    #         sens, spec = test_images(model, test_indices, labels, filenames, test_run, save_imgs=False)
    #     print("sensitivity:", sens)
    #     print("specificity:", spec)
    #
    #     sensitivities.append(sens)
    #     specificities.append(spec)
    #
    # print('scores:', test_scores)
    # print('average score', np.mean(test_scores))
    # print('average sensitivity', np.mean(sensitivities))
    # print('average specificity', np.mean(specificities))
    #
    #
    # print(model.metrics_names)
    #
    # print(hist.history.keys())
    #
    # epoch_num = range(len(hist.history['acc']))
    # train_error = np.subtract(1,np.array(hist.history['acc']))
    # test_error  = np.subtract(1,np.array(hist.history['val_acc']))
    #
    # plt.clf()
    # plt.plot(epoch_num, train_error, label='Training Error')
    # plt.plot(epoch_num, test_error, label="Validation Error")
    # plt.legend(shadow=True)
    # plt.xlabel("Training Epoch Number")
    # plt.ylabel("Error")
    # plt.savefig('results.png')
    # plt.close()
