import nibabel as nib
import argparse as ap

import numpy as np
import os

from keras.models import load_model
import keras.backend as K

if __name__ == "__main__":

    parser = ap.ArgumentParser(description="Tests an MRI image for motion and other artifacts and returns a probability of success.")

    parser.add_argument("t1image")
    args, leftovers = parser.parse_known_args()

    image_file = args.t1image
    print('Input image file:', image_file)

    # K.set_image_dim_ordering('th')

    try:
        img = nib.load(image_file).get_data()

        # img = resize_image_with_crop_or_pad(img, (168, 256, 244), mode='constant')
        # img = normalise_zero_one(img)

        img = np.subtract(img, np.min(img))
        img = np.divide(img, np.max(img))
        img = np.float32(img)

        (x_size, y_size, z_size) = img.shape
        # print('original image size:', img.shape)

        y_max, z_max = 256, 244

        while y_size < y_max or z_size < z_max:
            img = np.pad(img, 1, 'constant')
            (x_size, y_size, z_size) = img.shape

        y_mid, z_mid = y_size/2, z_size/2

        x_slice = int(x_size/2)
        y_start, y_stop = int(y_mid-y_max/2), int(y_mid+y_max/2)
        z_start, z_stop = int(z_mid-z_max/2), int(z_mid+z_max/2)

        # print('x_slice:', x_slice)
        # print('y_start, y_stop:', y_start, y_stop)
        # print('z_start, z_stop:', z_start, z_stop)

        # img_slice = img[x_slice, ][np.newaxis, ..., np.newaxis]
        # print(img_slice.shape)

        model = load_model(os.path.expanduser('~/IBIS-DeepMRIQC-model-v0.hdf5'))

        slice_predictions = []
        for i in range(10):
            slice_predictions.append(model.predict(img[x_slice-4+i, y_start:y_stop, z_start:z_stop][np.newaxis, ..., np.newaxis])[0][0])

        print(slice_predictions)
        print(np.mean(slice_predictions))

        try:
            K.clear_session()
        except:
            pass

    except Exception as e:
        print(e)
