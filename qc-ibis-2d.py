from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, Callback

import numpy as np
import h5py

import os, csv, time
import nibabel as nib

from dltk.core.io.preprocessing import normalise_zero_one, resize_image_with_crop_or_pad
# from custom_loss import sensitivity, specificity, true_positives, true_negatives, false_positives, false_negatives

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
target_size = (168, 256, 224)

train_indices, validation_indices, test_indices = [], [], []
results_dir = ''

def make_ibis_qc():
    with h5py.File(workdir + 'ibis.hdf5', 'w') as f:
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
                        pass_fail = 1
                    else:
                        one_hot = [1.0, 0.0]
                        pass_fail = 0

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
                    labels.append(pass_fail)

                    index += 1
                except Exception as e:
                    print('Error:', e)

        print('Total subjects we actually have:', index+1)

    return indices, labels


class SensSpec(Callback):

    def on_train_begin(self, logs={}):
        self.train_sens = []
        self.train_spec = []

        self.val_sens = []
        self.val_spec = []

    def on_epoch_end(self, batch, logs={}):

        train_sensitivity, train_specificity = sens_spec(train_indices, self.model)
        val_sensitivity, val_specificity = sens_spec(validation_indices, self.model)

        self.train_sens.append(train_sensitivity)
        self.train_spec.append(train_specificity)
        self.val_sens.append(val_sensitivity)
        self.val_spec.append(val_specificity)


    def on_train_end(self, logs={}):
        epoch_num = range(len(self.train_sens))

        plt.close()
        # plt.plot(epoch_num, hist.history['acc'], label='Training Accuracy')
        # plt.plot(epoch_num, hist.history['val_acc'], label="Validation Accuracy")
        plt.plot(epoch_num, self.train_sens, label='Train Sensitivity')
        plt.plot(epoch_num, self.train_spec, label='Train Specificity')
        plt.plot(epoch_num, self.val_sens, label='Validation Sensitivity')
        plt.plot(epoch_num, self.val_spec, label='Val Specificity')

        plt.legend(shadow=True)
        plt.xlabel("Training Epoch Number")
        plt.ylabel("Metric Value")
        plt.savefig(results_dir + 'training_metrics.png', bbox_inches='tight')
        plt.close()


def sens_spec(indices, model):
    with h5py.File(workdir + 'ibis.hdf5') as f:
        images = f['ibis_t1']
        labels = f['qc_label']

        predictions = np.zeros((len(indices)))
        actual = np.zeros((len(indices)))

        predict_batch = np.zeros((1, target_size[1], target_size[2], 1))

        for i, index in enumerate(indices):
            predict_batch[0, :, :, 0] = images[index, target_size[0] // 2, :, :]

            prediction = model.predict_on_batch(predict_batch)[0][0]
            if prediction >= 0.5:
                predictions[i] = 1
            else:
                predictions[i] = 0
            actual[i] = np.argmax(labels[index, ...])

        conf = confusion_matrix(actual, predictions)

        tp = conf[0][0]
        tn = conf[1][1]
        fp = conf[0][1]
        fn = conf[1][0]

        sensitivity = float(tp) / (float(tp) + float(fn) + 1e-10)
        specificity = float(tn) / (float(tn) + float(fp) + 1e-10)

    return sensitivity, specificity

def qc_model():
    nb_classes = 2

    conv_size = (3, 3)

    model = Sequential()

    model.add(Conv2D(16, conv_size, activation='relu', input_shape=(target_size[1], target_size[2], 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.1))

    model.add(Conv2D(32, conv_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.1))

    model.add(Conv2D(32, conv_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))

    model.add(Conv2D(64, conv_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.2))

    model.add(Conv2D(64, conv_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.3))

    model.add(Conv2D(128, conv_size, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.4))

    # model.add(Conv2D(256, conv_size, activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(nb_classes, activation='softmax', name='predictions'))

    return model

def batch(indices, n, random_slice=False):
    with h5py.File(workdir + 'ibis.hdf5', 'r') as f:
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


def plot_graphs(hist, results_dir, fold_num):
    epoch_num = range(len(hist.history['acc']))

    plt.clf()
    plt.plot(epoch_num, hist.history['acc'], label='Training Accuracy')
    plt.plot(epoch_num, hist.history['val_acc'], label="Validation Accuracy")
    # plt.plot(epoch_num, hist.history['sensitivity'], label='Train Sens')
    # plt.plot(epoch_num, hist.history['val_sensitivity'], label='Val Sens')
    # plt.plot(epoch_num, hist.history['specificity'], label='Train Spec')
    # plt.plot(epoch_num, hist.history['val_specificity'], label='Val Spec')

    plt.legend(shadow=True)
    plt.xlabel("Training Epoch Number")
    plt.ylabel("Metric Value")
    plt.savefig(results_dir + 'training_metrics_fold' + str(fold_num) + '.png', bbox_inches='tight')
    plt.close()

def predict_and_visualize(model, indices, results_dir):
    with h5py.File(workdir + 'ibis.hdf5', 'r') as f:
        images = f['ibis_t1']
        labels = f['qc_label']
        filenames = f['filename']

        predictions = []
        avg_pass = np.zeros((target_size[1], target_size[2], 3), dtype='float32')
        avg_fail = np.zeros((target_size[1], target_size[2], 3), dtype='float32')

        pass_imgs = 0.0
        fail_imgs = 0.0

        with open(results_dir + 'test_images.csv', 'w') as output_file:
            output_writer = csv.writer(output_file)
            output_writer.writerow(['Filename', 'Pass Probability', 'Actual'])

            for index in indices:
                img = images[index, target_size[0]//2, ...][np.newaxis, ..., np.newaxis]
                label = labels[index, ...]

                prediction = model.predict(img, batch_size=1)
                print('index:', index, 'probs:', prediction[0])

                output_writer.writerow([filenames[index, ...][2:-1], prediction[0][1], np.argmax(label)])

                predictions.append(np.argmax(prediction[0]))

            model.compile(loss='categorical_crossentropy',
                          optimizer='sgd',
                          metrics = ["accuracy"])
            layer_idx = utils.find_layer_idx(model, 'predictions')
            model.layers[layer_idx].activation = activations.linear
            model = utils.apply_modifications(model)

        for i, (index, prediction) in enumerate(zip(indices, predictions)):
            fig, ax = plt.subplots(1, 2, figsize=(12, 8))

            actual = np.argmax(labels[index, ...])
            print('actual, predicted PASS/FAIL:', actual, prediction)
            if prediction == actual:
                decision = '_right_'
            else:
                decision = '_wrong_'

            if actual == 1:
                qc_status = 'PASS'
            else:
                qc_status = 'FAIL'

            filename = qc_status + decision + str(filenames[index, ...])[2:-1][:-4] + '.png'
            # filename = str(i) + decision + qc_status + '.png'

            plt.suptitle(filename)

            img = images[index, target_size[0] // 2, ...][np.newaxis, ..., np.newaxis]

            ax[0].imshow(img[0, ..., 0], cmap='gray')
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[0].set_xlabel('Input Image')

            for j, type in enumerate(['guided']):
                grads = visualize_cam(model, layer_idx, filter_indices=prediction, seed_input=img[0, ...], backprop_modifier=type)
                # print('gradient shape:', grads.shape)

                heatmap = np.uint8(cm.jet(grads)[:,:,0,:3]*255)
                gray = np.uint8(img[0, :, :, :]*255)
                gray3 = np.dstack((gray,)*3)

                img_ax = ax[j+1].imshow(overlay(heatmap, gray3, alpha=0.2))
                ax[j+1].set_xticks([])
                ax[j+1].set_yticks([])
                plt.colorbar(img_ax)
                ax[j+1].set_xlabel('Decision Regions (Guided Grad-CAM)')

                if prediction == 0:
                    avg_fail += heatmap
                    fail_imgs += 1.0
                else:
                    avg_pass += heatmap
                    pass_imgs += 1.0

            plt.subplots_adjust()
            plt.savefig(results_dir + filename, bbox_inches='tight')
            plt.close()

        # pass_regions = np.divide(avg_pass, pass_imgs)
        # fail_regions = np.divide(avg_fail, fail_imgs)

        plt.figure()
        plt.imshow(avg_pass, vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(results_dir + 'average_pass_gradient.png', bbox_inches='tight')
        plt.figure()
        plt.imshow(avg_fail, vmin=0, vmax=1)
        plt.axis('off')
        plt.savefig(results_dir + 'average_fail_gradient.png', bbox_inches='tight')
        plt.close()

def verify_hdf5(indices, results_dir):
    with h5py.File(workdir + 'ibis.hdf5', 'r') as f:
        images = f['ibis_t1']
        labels = f['qc_label']
        filenames = f['filename']

        for index in indices:
            img = images[index, target_size[0]//2, :, :]
            label = labels[index, ...]
            filename = filenames[index, ...]

            plt.imshow(img, cmap='gray')
            plt.xlabel(str(label))
            plt.ylabel(str(filename[2:-1]))
            plt.savefig(results_dir + 'img-' + str(index), bbox_inches='tight')
            plt.close()


if __name__ == "__main__":
    start_time = time.time()

    batch_size = 16

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

    labels = np.asarray(labels, dtype='uint8')
    indices = np.asarray(indices, dtype='uint8')

    print('indices', np.asarray(indices))
    print('labels', np.asarray(labels))

    skf = StratifiedKFold(n_splits=10)

    sgd = SGD(lr=1e-3, momentum=0.9, decay=1e-6, nesterov=True)
    adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)

    score_metrics = ["accuracy"]

    model = qc_model()
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=score_metrics)

    scores = {}
    scores['train_acc'] = []
    scores['val_acc'] = []
    scores['test_acc'] = []
    scores['train_sens'] = []
    scores['train_spec'] = []
    scores['val_sens'] = []
    scores['val_spec'] = []
    scores['test_sens'] = []
    scores['test_spec'] = []

    for k, (train_indices, test_indices) in enumerate(skf.split(np.asarray(indices), labels)):

        results_dir = workdir + '/experiment-' + str(experiment_number) + '/fold-' + str(k) + '/'
        os.makedirs(results_dir)
        model = qc_model()

        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=score_metrics)

        validation_indices = test_indices[::2]
        test_indices = test_indices[1::2]

        print(labels[test_indices])
        print(np.sum(labels[test_indices]))
        print(len(labels[test_indices]))

        print('train indices:', len(train_indices), np.sum(labels[np.asarray(train_indices, dtype='uint8')]))
        print('validation indices:', len(validation_indices), np.sum(labels[np.asarray(validation_indices, dtype='uint8')]))
        print('test indices:', len(test_indices), np.sum(labels[np.asarray(test_indices, dtype='uint8')]))

        # verify_hdf5(reversed(train_indices), results_dir)

        f = h5py.File(workdir + 'ibis.hdf5', 'r')
        h5labels = f['qc_label']
        for index in train_indices:
            print(index, labels[index], h5labels[index, ...])
        f.close()

        model_checkpoint = ModelCheckpoint(results_dir + "best_weights" + "_fold_" + str(k) + ".hdf5", monitor="val_acc", verbose=0, save_best_only=True, save_weights_only=False, mode='max')

        hist = model.fit_generator(batch(train_indices, batch_size, True), np.ceil(len(train_indices)/batch_size), epochs=400, validation_data=batch(validation_indices, batch_size), validation_steps=np.ceil(len(validation_indices)//batch_size), callbacks=[model_checkpoint], class_weight = {0:10, 1:1})

        model.load_weights(results_dir + "best_weights" + "_fold_" + str(k) + ".hdf5")
        model.save(results_dir + 'ibis_qc_model' + str(k) + '.hdf5')

        train_metrics = model.evaluate_generator(batch(train_indices, batch_size, True), np.ceil(len(train_indices)/batch_size))
        val_metrics = model.evaluate_generator(batch(validation_indices, batch_size, True), np.ceil(len(validation_indices)/batch_size))
        test_metrics = model.evaluate_generator(batch(test_indices, batch_size, True), np.ceil(len(test_indices)/batch_size))

        print(model.metrics_names)
        print('train:', train_metrics, 'val:', val_metrics, 'test:', test_metrics)

        plot_graphs(hist, results_dir, k)

        train_sens, train_spec = sens_spec(train_indices, model)
        val_sens, val_spec = sens_spec(validation_indices, model)
        test_sens, test_spec = sens_spec(test_indices, model)

        scores['train_acc'].append(train_metrics[1])
        scores['val_acc'].append(val_metrics[1])
        scores['test_acc'].append(test_metrics[1])

        scores['train_sens'].append(train_sens)
        scores['train_spec'].append(train_spec)
        scores['val_sens'].append(val_sens)
        scores['val_spec'].append(val_spec)

        predict_and_visualize(model, test_indices, results_dir)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        model.save(results_dir + 'ibis_qc_model' + str(k) + '.hdf5')

    plt.close()

    score_data = []
    score_labels = []

    score_data.append(scores['train_acc'])
    score_labels.append('Training\nAccuracy')

    score_data.append(scores['val_acc'])
    score_labels.append('Validation\nAccuracy')

    score_data.append(scores['test_acc'])
    score_labels.append('Test\nAccuracy')

    score_data.append(scores['train_sens'])
    score_labels.append('Training\nSensitivity')

    score_data.append(scores['val_sens'])
    score_labels.append('Validation\nSensitivity')

    score_data.append(scores['test_sens'])
    score_labels.append('Test\nSensitivity')

    score_data.append(scores['train_spec'])
    score_labels.append('Training\nSpecificity')

    score_data.append(scores['val_spec'])
    score_labels.append('Validation\nSpecificity')

    score_data.append(scores['test_spec'])
    score_labels.append('Test\nSpecificity')

    bplot = plt.boxplot(score_data, patch_artist=True)
    plt.xticks(np.arange(len(score_data)), score_labels)

    # fill with colors
    colors = ['pink', 'red', 'darkred', 'pink', 'red', 'darkred', 'pink', 'red', 'darkred']

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.xlabel('Metric')
    plt.ylabel('Value')

    results_dir = workdir + '/experiment-' + str(experiment_number) + '/'
    plt.savefig(results_dir + 'metrics_boxplot.png')

    print(scores)

    print('time taken:', (time.time() - start_time) / 60, 'minutes')
    print('This experiment is brought to you by the number:', experiment_number)