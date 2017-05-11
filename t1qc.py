import os
import nibabel as nib
import argparse as ap

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten, BatchNormalization, SpatialDropout2D
from keras.optimizers import SGD

from keras import backend as K

def qc_model():
    nb_classes = 2

    model = Sequential()

    model.add(Conv2D(16, (3, 3), padding='same', input_shape=(1, 256, 224)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.2))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SpatialDropout2D(0.3))

    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SpatialDropout2D(0.3))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(SpatialDropout2D(0.3))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(SpatialDropout2D(0.3))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(SpatialDropout2D(0.4))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(SpatialDropout2D(0.5))

    model.add(Flatten())
    model.add(Dense(256, kernel_initializer='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, kernel_initializer='uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(nb_classes, kernel_initializer='uniform'))
    model.add(Activation('softmax'))

    sgd = SGD(lr=1e-3, momentum=0.9, decay=1e-6, nesterov=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=["accuracy"])

    return model


if __name__ == "__main__":

    parser = ap.ArgumentParser(description="Tests an MRI image for motion and other artifacts and returns a probability of success.")

    parser.add_argument("t1image")
    args, leftovers = parser.parse_known_args()

    image_file = args.t1image
    print('Input image file:', image_file)


    K.set_image_dim_ordering('th')

    try:
        img = nib.load(image_file).get_data()

        (x_size, y_size, z_size) = img.shape
        print('original image size:', img.shape)

        y_max, z_max = 256, 224

        while y_size < y_max or z_size < z_max:
            img = np.pad(img, 1, 'constant')
            (x_size, y_size, z_size) = img.shape

        y_mid, z_mid = y_size/2, z_size/2

        x_slice = int(x_size/2)
        y_start, y_stop = int(y_mid-y_max/2), int(y_mid+y_max/2)
        z_start, z_stop = int(z_mid-z_max/2), int(z_mid+z_max/2)

        print('x_slice:', x_slice)
        print('y_start, y_stop:', y_start, y_stop)
        print('z_start, z_stop:', z_start, z_stop)


        img_slice = img[x_slice, y_start:y_stop, z_start:z_stop][np.newaxis, np.newaxis, ...]

        print(img_slice.shape)

        model = Sequential()
        model = qc_model()
        model.load_weights('~/ibis-qc.hdf5')
        prediction = model.predict(img_slice)

        print(prediction[0][0])

        K.clear_session()

    except FileExistsError:
        print("File not found")


