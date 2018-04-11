import nibabel as nib
import argparse as ap

import numpy as np
import os

import torch
from torch.autograd import Variable

from train_ibis_qc import ConvolutionalQCNet

import onnx
from onnx_tf.backend import prepare

# taken from DLTK
def normalise_zero_one(image):
    """Image normalisation. Normalises image to fit [0, 1] range."""

    image = image.astype(np.float32)
    ret = (image - np.min(image))
    ret /= (np.max(image) + 0.000001)
    return ret

# taken from DLTK
def resize_image_with_crop_or_pad(image, img_size=(64, 64, 64), **kwargs):
    """Image resizing. Resizes image by cropping or padding dimension
     to fit specified size.
    Args:
        image (np.ndarray): image to be resized
        img_size (list or tuple): new image size
        kwargs (): additional arguments to be passed to np.pad
    Returns:
        np.ndarray: resized image
    """

    assert isinstance(image, (np.ndarray, np.generic))
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), 'Example size doesnt fit image size'

    # Get the image dimensionality
    rank = len(img_size)

    # Create placeholders for the new shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for dim in range(rank)]

    slicer = [slice(None)] * rank

    # For each dimensions find whether it is supposed to be cropped or padded
    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        # Create slicer object to crop or leave each dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    # Pad the cropped image to extend the missing dimension
    return np.pad(image[slicer], to_padding, **kwargs)


def qc_image(image, target_size=(160, 256, 224), model_version=None, using_onnx=False):

    if model_version == None:
        model_version = 1

    start_slice = (target_size[0] // 2) - 5
    end_slice = (target_size[0] // 2) + 5

    slices = image[start_slice:end_slice, :, :][..., np.newaxis]

    if using_onnx:
        model_path = os.path.expanduser('~/ibis_qc_net_v' + str(model_version) + '.onnx')

        model = onnx.load(model_path)
        tf_rep = prepare(model)

        print(tf_rep.predict_net)
        print('-----')
        print(tf_rep.input_dict)
        print('-----')
        print(tf_rep.uninitialized)

        predictions = tf_rep.run(slices)._0

    else:
        model_path = os.path.expanduser('~/ibis_qc_net_v' + str(model_version) + '.tch')

        model = ConvolutionalQCNet()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        slices_tensor = torch.Tensor(slices)
        data = Variable(slices_tensor, volatile=True)
        output = model(data)

        predicted_tensors = output.data.cpu().numpy()
        print('predicted_tensors:', predicted_tensors)

    print('Predictions:', predictions)
    print('Variance:', np.var(predictions[:, 1]))

    prediction = np.sum(predictions[:, 1])
    confidence = 1 - np.var(predictions[:, 1])

    return prediction, confidence


def preprocess_image(image, target_size=(160, 256, 224), preprocessing_version=None):

    if preprocessing_version == 1:
        resized_image = resize_image_with_crop_or_pad(img, img_size=target_size, mode='constant')
        normalized_image = normalise_zero_one(resized_image)

    return normalized_image

if __name__ == "__main__":

    parser = ap.ArgumentParser(description="Tests an MRI image for motion and other artifacts and returns a probability of success.")

    parser.add_argument("t1image")

    args, leftovers = parser.parse_known_args()

    image_file = args.t1image

    target_size = (160, 256, 224)


    print('Input image file:', image_file)


    img = nib.load(image_file).get_data()

    preprocessed_image = preprocess_image(img, target_size, preprocessing_version=1)
    prediction, confidence = qc_image(preprocessed_image, target_size=target_size, model_version=1)

    print(prediction, confidence)
