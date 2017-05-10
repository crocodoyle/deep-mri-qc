import os
import nibabel as nib
import argparse as ap

from keras.models import Sequential


if __name__ == "__main__":

    parser = ap.ArgumentParser(description="Tests an MRI image for motion and other artifacts and returns a probability of success.")

    parser.add_argument("t1image")
    args, leftovers = parser.parse_known_args()

    image_file = args.t1image
    print('Input image file:', image_file)

    try:
        img = nib.load(image_file).get_data()

        (x_size, y_size, z_size) = img.shape

        x_max = 256
        y_max = 224
                
        #TODO: pad and crop image
        if x_size > x_max:
            pass

        if y_size > y_max:
            pass

        if x_size < x_max:
            pass

        if y_size < y_max:
            pass

        slice = img[:, y_size, z_size / 2]



        image_size = (244, 256)

    except FileExistsError:
        print("File not found")


    model = Sequential()
    model.load_weights('models/ibis-qc.hdf5')
    model.predict()