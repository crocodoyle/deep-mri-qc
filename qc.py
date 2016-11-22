from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

import numpy as np

import os
import nibabel

import matplotlib.pyplot as plt

from sklearn.cross_validation import StratifiedShuffleSplit

def loadData(fail_path, pass_path):
    print "loading data..."
    images = []
    labels = []
    filenames = []
    max_images = 120
    i = 0
    for root, dirs, files in os.walk(fail_path, topdown=False):
        for name in files:
            images.append(nibabel.load(os.path.join(root, name)).get_data()[:,:,120])
            labels.append([1, 0])
            filenames.append(os.path.join(root, name))
        i += 1
        if i >= max_images:
            break
#            plt.imshow(images[-1][:,:, 120])
#            print(os.path.join(root, name))

    i=0
    for root, dirs, files in os.walk(pass_path, topdown=False):
        for name in files:
            images.append(nibabel.load(os.path.join(root, name)).get_data()[:,:,120])
            labels.append([0, 1])
            filenames.append(os.path.join(root, name))
        i += 1
        if i >= max_images:
            break
#            plt.imshow(images[-1][:,:, 120])
#            print(os.path.join(root, name))

    print np.shape(np.asarray(images))
    indices = StratifiedShuffleSplit(labels, test_size=0.4, n_iter=1, random_state=None)
    test_index = None
    for train_indices, test_indices in indices:
        x_train = np.asarray(images)[train_indices]
        x_test = np.asarray(images)[test_indices]
        y_train = np.asarray(labels)[train_indices]
        y_test = np.asarray(labels)[test_indices]
        
        test_index = test_indices        


    filename_test = []
    for i, f in enumerate(filenames):
        if i in test_index:
            filename_test.append(f)

    x_train = np.reshape(x_train, (np.shape(x_train)[0], 160*256))
    x_test = np.reshape(x_test, (np.shape(x_test)[0], 160*256))

    return x_train, x_test, y_train, y_test, filename_test

    
def model_train(x_train, x_test, y_train, y_test, filename_test):
    
    print "shape of training data:", np.shape(x_train)
    print "shape of testing data:", np.shape(x_test)
    print "shape of training labels:", np.shape(y_train)
    print "shape of testing labels:", np.shape(y_test)
    print "filename list:", len(filename_test)
    
    data_dim = 160*256
    nb_classes = 2
    
    model = Sequential()
    
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(1024, input_dim=data_dim, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(256, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, init='uniform'))
    model.add(Activation('softmax'))
    
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',  
                  metrics=["accuracy"])
    
    model.fit(x_train, y_train,
              nb_epoch=20,
              batch_size=16)
    
    score = model.evaluate(x_test, y_test, batch_size=16)
    print model.metrics_names
    print score
    
    for i in range(len(x_test)):
        test_case = x_test[i,:]
        label = y_test[i]

        test_case = np.reshape(test_case, (1, len(test_case)))
        predictions = model.predict(test_case, batch_size=1)
        image = np.reshape(test_case, (160,256))
        plt.imshow(image.T)
        plt.show()
        print "predictions:", predictions
        print "label:", label
#        print "file:", filename_test[i]
        
if __name__ == "__main__":
    print "Running automatic QC"
    fail_data = "/home/adoyle/T1_Minc_Fail"
    pass_data = "/home/adoyle/T1_Minc_Pass"
    
    x_train, x_test, y_train, y_test, filename_test = loadData(fail_data, pass_data)
    
    model_train(x_train, x_test, y_train, y_test, filename_test)
    #chooo chooooo
