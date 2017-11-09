from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint

import numpy as np
import h5py

import os, csv, time
import nibabel as nib

from dltk.core.io.preprocessing import normalise_zero_one, resize_image_with_crop_or_pad
from custom_loss import sensitivity, specificity

from collections import defaultdict

import pickle as pkl

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import confusion_matrix

from vis.visualization import visualize_cam, overlay
from vis.utils import utils
from keras import activations

# from vis.utils import find_layer_idx

workdir = '/home/users/adoyle/deepqc/IBIS/'
datadir = '/data1/users/adoyle/IBIS/'

label_file = datadir + 't1_ibis_QC_labels.csv'

total_subjects = 2020
target_size = (168, 256, 244)

def make_ibis_qc():
    f = h5py.File(workdir + 'ibis.hdf5', 'w')
    f.create_dataset('ibis_t1', (total_subjects, target_size[0], target_size[1], target_size[2]), dtype='float32')
    f.create_dataset('qc_label', (total_subjects, 2), dtype='float32')
    dt = h5py.special_dtype(vlen=bytes)
    f.create_dataset('filename', (total_subjects, ), dtype=dt)

    index = 0

    indices = []
    labels = []

    with open(label_file, 'r') as labels_csv:
        qc_reader = csv.reader(labels_csv)
        next(qc_reader)

        for line in qc_reader:
            try:
                t1_filename = line[3][9:]
                label = line[4]

                if 'Pass' in label:
                    one_hot = [0.0, 1.0]
                else:
                    one_hot = [1.0, 0.0]

                f['qc_label'][index, :] = one_hot
                t1_data = nib.load(datadir + t1_filename).get_data()

                if not t1_data.shape == target_size:
                    # print('resizing from', t1_data.shape)
                    t1_data = resize_image_with_crop_or_pad(t1_data, img_size=target_size, mode='constant')

                f['ibis_t1'][index, ...] = normalise_zero_one(t1_data)
                f['filename'][index] = t1_filename.split('/')[-1]

                # plt.imshow(t1_data[96, ...])
                # plt.axis('off')
                # plt.savefig(output_dir + t1_filename[:-4] + '.png', bbox_inches='tight', cmap='gray')

                indices.append(index)
                labels.append(np.argmax(one_hot))

                index += 1
            except Exception as e:
                print('Error:', e)

    print('Total subjects we actually have:', index+1)
    f.close()

    return indices, labels


def qc_model():
    nb_classes = 2

    conv_size = (3, 3)

    model = Sequential()

    model.add(Conv2D(16, conv_size, activation='relu', input_shape=(target_size[1], target_size[2], 1)))
    model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=pool_size))
    # model.add(Dropout(0.2))

    model.add(Conv2D(32, conv_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.3))

    model.add(Conv2D(32, conv_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.3))

    model.add(Conv2D(64, conv_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.3))

    model.add(Conv2D(64, conv_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.3))

    model.add(Conv2D(128, conv_size, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.4))

    model.add(Conv2D(256, conv_size, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(nb_classes, activation='softmax', name='predictions'))

    sgd = SGD(lr=1e-3, momentum=0.9, decay=1e-6, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  #metrics=["accuracy", sensitivity, specificity])
                  metrics = ["accuracy"])

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
            x_train[i%n, :, :, 0] = images[index, target_size[0]//2+rn, :, :]
            y_train[i%n, :]   = labels[index, ...]
            samples_this_batch += 1
            if (i+1) % n == 0:
                yield (x_train, y_train)
                samples_this_batch = 0
            elif i == len(indices)-1:
                yield (x_train[0:samples_this_batch, ...], y_train[0:samples_this_batch, :])


def test_images(model, test_indices, save_imgs=True):
    f = h5py.File(workdir + 'ibis.hdf5', 'r')
    images = f['ibis_t1']
    labels = f['qc_label']
    filename_test = f['filenames']

    predictions = np.zeros((len(test_indices)))
    actual = np.zeros((len(test_indices)))

    predict_batch = np.zeros((1, target_size[1], target_size[2], 1))

    print("test indices:", len(test_indices))
    print("test index max:", max(test_indices))
    print("labels:", len(labels))
    print("filenames:", len(filename_test))

    for i, index in enumerate(test_indices):
        predict_batch[0,:,:,0] = images[index,target_size[0]//2, :,:]

        prediction = model.predict_on_batch(predict_batch)[0][0]
        if prediction >= 0.5:
            predictions[i] = 1
        else:
            predictions[i] = 0
        actual[i] = labels[index,0]

        if save_imgs:
            plt.imshow(images[index,target_size[0]//2+10,:,:], cmap='gray')
            if predictions[i] == 1 and actual[i] == 1:
                plt.savefig(results_dir + 'fail_right_' + os.path.basename(filename_test[i]) + ".png")
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

def plot_graphs(hist, results_dir, fold_num):
    epoch_num = range(len(hist.history['acc']))

    plt.clf()
    plt.plot(epoch_num, hist.history['acc'], label='Training Accuracy')
    plt.plot(epoch_num, hist.history['val_acc'], label="Validation Accuracy")
    # plt.plot(epoch_num, hist.history['sensitivity'], label='Training Sensitivity')
    # plt.plot(epoch_num, hist.history['val_sensitivity'], label='Validation Sensitivity')
    # plt.plot(epoch_num, hist.history['specificity'], label='Training Specificity')
    # plt.plot(epoch_num, hist.history['val_specificity'], label='Validation Specificity')

    plt.legend(shadow=True)
    plt.xlabel("Training Epoch Number")
    plt.ylabel("Metric Value")
    plt.savefig(results_dir + 'training_metrics_fold' + str(fold_num) + '.png', bbox_inches='tight')
    plt.close()

def predict_and_visualize(model, indices, results_dir):
    f = h5py.File(workdir + 'ibis.hdf5', 'r')
    images = f['ibis_t1']
    labels = f['qc_label']
    filenames = f['filename']

    predictions = []

    with open(results_dir + 'test_images.csv', 'w') as output_file:
        output_writer = csv.writer(output_file)
        output_writer.writerow(['Filename', 'Probability'])

        for index in indices:
            img = images[index, target_size[0]//2, ...][np.newaxis, ..., np.newaxis]
            label = labels[index, ...]

            prediction = model.predict(img, batch_size=1)
            print('probs:', prediction[0])

            output_writer.writerow([filenames[index, ...], prediction[0][0], np.argmax(label)])

            predictions.append(np.argmax(prediction[0]))


    for i, (index, prediction) in enumerate(zip(indices, predictions)):

        layer_idx = utils.find_layer_idx(model, 'predictions')
        model.layers[layer_idx].activation = activations.linear
        model = utils.apply_modifications(model)

        grads = visualize_cam(model, layer_idx, filter_indices=prediction, seed_input=img[0, ...], backprop_modifier='guided')

        heatmap = np.uint8(cm.jet(grads)[:,:,0,:3]*255)
        gray = np.uint8(cm.gray(np.hstack((img[0, :, :, 0],)*3)*255))[..., :3]

        print('image shape, heatmap shape', gray.shape, heatmap.shape)

        plt.imshow(overlay(heatmap, gray))

        actual = np.argmax(labels[index, ...])
        if prediction == actual:
            decision = '_right_'
        else:
            decision = '_wrong_'

        if actual == 1:
            qc_status = 'PASS'
        else:
            qc_status = 'FAIL'

        # filename = qc_status + decision + filenames[index, ...][:-4] + '.png'
        filename = str(i) + decision + qc_status + '.png'

        plt.axis('off')
        plt.savefig(results_dir + filename, bbox_inches='tight')
        plt.clf()

    f.close()

if __name__ == "__main__":
    start_time = time.time()

    batch_size = 32

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

    remake = False
    if remake:
        indices, labels = make_ibis_qc()
        pkl.dump(indices, open(workdir + 'valid_indices.pkl', 'wb'))
        pkl.dump(labels, open(workdir + 'qc_labels.pkl', 'wb'))
    else:
        indices = pkl.load(open(workdir + 'valid_indices.pkl', 'rb'))
        labels = pkl.load(open(workdir + 'qc_labels.pkl', 'rb'))

    print('indices', indices)
    print('labels', labels)

    skf = StratifiedKFold(n_splits=4)

    model = qc_model()
    model.summary()

    scores = {}
    for metric in model.metrics_names:
        scores[metric] = []

    for k, (train_indices, test_indices) in enumerate(skf.split(np.asarray(indices), np.asarray(labels))):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        result_indices = sss.split(np.asarray(test_indices), np.asarray(labels)[test_indices])

        test_indices, validation_indices = next(result_indices)
        print('train indices:', train_indices)
        print('validation indices:', validation_indices)
        print('test indices:', test_indices)

        model_checkpoint = ModelCheckpoint(results_dir + "best_weights" + "_fold_" + str(k) + ".hdf5", monitor="val_acc", verbose=0, save_best_only=True, save_weights_only=False, mode='max')

        hist = model.fit_generator(batch(train_indices, batch_size, True), len(train_indices)//batch_size, epochs=5, validation_data=batch(validation_indices, batch_size), validation_steps=len(validation_indices)//batch_size+1, callbacks=[model_checkpoint], class_weight = {0:.7, 1:.3})

        model.load_weights(results_dir + "best_weights" + "_fold_" + str(k) + ".hdf5")
        model.save(results_dir + 'ibis_qc_model' + str(k) + '.hdf5')

        metrics = model.evaluate_generator(batch(test_indices, batch_size, True), len(test_indices)//32+1)

        print(model.metrics_names)
        print(metrics)

        plot_graphs(hist, results_dir, k)

        for metric_name, score in zip(model.metrics_names, metrics):
            scores[metric_name].append(score)

        predict_and_visualize(model, test_indices, results_dir)

    print(metric, scores[metric])
    for metric in model.metrics_names:
        print(metric, np.mean(scores[metric]))

    print(scores)

    print('time taken:', (time.time() - start_time) / 60, 'minutes')
    print('This experiment is brought to you by the number:', experiment_number)