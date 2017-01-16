from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, BatchNormalization, SpatialDropout2D
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils.visualize_util import plot

import numpy as np
import h5py

import os
import nibabel

import cPickle as pkl

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix


images_dir = '/gs/scratch/adoyle/'
cluster = False

if cluster:
    images_dir  = '/gs/scratch/adoyle/'
    scratch_dir = os.environ.get('RAMDISK') + '/'
else:
    images_dir   = '/home/adoyle/'
    scratch_dir  = images_dir

print 'SCRATCH', scratch_dir
print 'IMAGES:', images_dir


def load_data(fail_path, pass_path):
    print "loading data..."
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
               print np.shape(img)
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

    indices = StratifiedShuffleSplit(labels, test_size=0.3, n_iter=1, random_state=None)

    train_index, test_index = None, None
    for train_indices, test_indices in indices:
        train_index = train_indices
        test_index  = test_indices

    # pkl.dump(labels, images_dir + 'labels.pkl')

    return train_index, test_index, labels, filenames

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

    model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=(1, 256, 224)))
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

    model.add(Dense(nb_classes, init='uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=1e-3, momentum=0.9, decay=1e-6, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=["accuracy"])

    return model

def model_train(x_train, x_test, y_train, y_test, filename_test):

    print "shape of training data:", np.shape(x_train)
    print "shape of testing data:", np.shape(x_test)
    print "shape of training labels:", np.shape(y_train)
    print "shape of testing labels:", np.shape(y_test)
    print "filename list:", len(filename_test)

#    data_dim = 160*256


    model.fit(x_train, y_train,
              nb_epoch=200,
              batch_size=50)
    #should return model to workspace so that I can keep training it

    score = model.evaluate(x_test, y_test, batch_size=10)
    print model.metrics_names
    print score

    for i in range(len(x_test)):
        test_case = x_test[i,...]
        label = y_test[i]

        test_case = np.reshape(test_case, (1, 1, np.shape(test_case)[1], np.shape(test_case)[2]))
        predictions = model.predict(test_case, batch_size=1)
        image = np.reshape(test_case[0, 1,...], (256, 224))
        # plt.imshow(image.T)
        # plt.show()
        print "predictions:", predictions
        print "label:", label
#        print "file:", filename_test[i]

def batch(indices, labels, n, random_slice=False):
    f = h5py.File(scratch_dir + 'ibis.hdf5', 'r')
    images = f.get('ibis_t1')

    x_train = np.zeros((n, 1, 256, 224), dtype=np.float32)
    y_train = np.zeros((n, 2), dtype=np.int8)

    while True:
        np.random.shuffle(indices)

        samples_this_batch = 0
        for i, index in enumerate(indices):
            if random_slice:
                rn=np.random.randint(-4,4)
            else:
                rn=0
            x_train[i%n, 0, :, :] = images[index, 80+rn, :, :]
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

    predict_batch = np.zeros((1, 1, 256, 224))

    print "test indices:", len(test_indices)
    print "test index max:", max(test_indices)
    print "labels:", len(labels)
    print "filenames:", len(filename_test)

    for i, index in enumerate(test_indices):
        predict_batch[0,0,:,:] = images[index, 80+slice_modifier,:,:]

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
                plt.savefig('/home/adoyle/images/pass_right' + os.path.basename(filename_test[i]) + '.png')
            elif predictions[i] == 1 and actual[i] == 0:
                plt.savefig('/home/adoyle/images/pass_wrong_' + os.path.basename(filename_test[i]) + '.png')
            elif predictions[i]  == 0 and actual[i] == 1:
                plt.savefig('/home/adoyle/images/fail_wrong_' + os.path.basename(filename_test[i]) + '.png')
            plt.clf()

    conf = confusion_matrix(actual, predictions)
    print 'Confusion Matrix'
    print conf

    print np.shape(conf)

    tn = conf[0][0]
    tp = conf[1][1]
    fn = conf[0][1]
    fp = conf[1][0]

    print 'true negatives:', tn
    print 'true positives:', tp
    print 'false negatives:', fn
    print 'false positives:', fp

    sensitivity = float(tp) / (float(tp) + float(fn))
    specificity = float(tn) / (float(tn) + float(fp))


    print 'sens:', sensitivity
    print 'spec:', specificity

    return sensitivity, specificity

if __name__ == "__main__":
    print "Running automatic QC"
    fail_data = images_dir + "T1_Minc_Fail"
    pass_data = images_dir + "T1_Minc_Pass"

    train_indices, test_indices, labels, filenames = load_data(fail_data, pass_data)

    model = qc_model()
    model.summary()
    plot(model, to_file="model.png")

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4, min_lr=0.001)
    stop_early = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
    model_checkpoint = ModelCheckpoint("models/best_model.hdf5", monitor="val_acc", verbose=0, save_best_only=True, save_weights_only=False, mode='auto')

    hist = model.fit_generator(batch(train_indices, labels, 2,True), nb_epoch=400, samples_per_epoch=len(train_indices), validation_data=batch(test_indices, labels, 2), nb_val_samples=len(test_indices), callbacks=[model_checkpoint], class_weight = {0:.7, 1:.3})


    model.load_weights('models/best_model.hdf5')


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
        print "sensitivity:", sens
        print "specificity:", spec

        sensitivities.append(sens)
        specificities.append(spec)

    print 'scores:', test_scores
    print 'average score', np.mean(test_scores)
    print 'average sensitivity', np.mean(sensitivities)
    print 'average specificity', np.mean(specificities)


    print model.metrics_names

    print hist.history.keys()

    epoch_num = range(len(hist.history['acc']))
    train_error = np.subtract(1,np.array(hist.history['acc']))
    test_error  = np.subtract(1,np.array(hist.history['val_acc']))

    plt.clf()
    plt.plot(epoch_num, train_error, label='Training Error')
    plt.plot(epoch_num, test_error, label="Testing Error")
    plt.legend(shadow=True)
    plt.xlabel("Training Epoch Number")
    plt.ylabel("Error")
    plt.savefig('results.png')
    plt.close()

    model.save('conv-2d.hdf5')
