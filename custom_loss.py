# Author: Alexandre Hutton (ahutton@cim.mcgill.ca)

from keras import backend as K
import numpy as np

import tensorflow as tf

def dice_loss(y_true, y_pred):
    """
    Computes approximate DICE coefficient as a loss by using the negative, computed with the Keras backend. The overlap\
     and total are offset to prevent 0/0, and the values are not rounded in order to keep the gradient information.
    Args:
        :arg y_true: Ground truth
        :arg y_pred: Predicted value for some input

    Returns
        :return: Approximate DICE coefficient.
    """
    ytf = K.flatten(y_true)
    ypf = K.flatten(y_pred)

    overlap = K.sum(ytf*ypf)
    total = K.sum(ytf*ytf) + K.sum(ypf * ypf)
    return -(2*overlap + K.epsilon()) / (total + K.epsilon())


def dice_metric(y_true, y_pred):
    """
    Computes DICE coefficient, computed with the Keras backend.
    Args:
        :arg y_true: Ground truth
        :arg y_pred: Predicted value for some input

    Returns
        :return: DICE coefficient
    """
    ytf = K.round(K.flatten(y_true))
    ypf = K.round(K.flatten(y_pred))

    overlap = 2*K.sum(ytf*ypf)
    total = K.sum(ytf*ytf) + K.sum(ypf * ypf)

    return overlap / total


def dice_np(im1, im2):
    """
        Computes DICE coefficient, computed with the Numpy
        Args:
            :arg im1: First image
            :arg im2: Second image

        Returns
            :return: DICE coefficient
            :rtype: float
        """
    im3 = np.round(np.ndarray.flatten(im1))
    im4 = np.round(np.ndarray.flatten(im2))

    overlap = 2*np.dot(im3, im4)
    total = np.dot(im3, im3) + np.dot(im4, im4)
    return overlap / total


def true_positives(y_true, y_pred):
    """Return number of true positives"""
    predictions = K.argmax(y_pred)
    truth = K.cast(K.argmax(y_true), dtype='bool')

    positive_pred = K.equal(predictions, 1)

    return K.cast(K.equal(K.equal(truth, positive_pred), K.equal(truth, True)), dtype='float32')

    # return K.cast(K.equal(positive_true, positive_pred), dtype='float32')


def true_negatives(y_true, y_pred):
     """Return number of true negatives"""
     predictions = K.argmax(y_pred)
     truth = K.cast(K.argmax(y_true), dtype='bool')

     negative_pred = K.equal(predictions, 0)

     return K.cast(K.equal(K.equal(truth, negative_pred), K.equal(truth, True)), dtype='float32')


def false_positives(y_true, y_pred):
    """Return number of false positives"""
    predictions = K.argmax(y_pred)
    truth = K.cast(K.argmax(y_true), dtype='bool')

    positive_pred = K.equal(predictions, 1)

    return K.cast(K.equal(K.not_equal(truth, positive_pred), K.equal(truth, False)), dtype='float32')


def false_negatives(y_true, y_pred):
    """Return number of false negatives"""
    predictions = K.argmax(y_pred)
    truth = K.cast(K.argmax(y_true), dtype='bool')

    negative_pred = K.equal(predictions, 0)

    return K.cast(K.equal(K.not_equal(truth, negative_pred), K.equal(truth, True)), dtype='float32')


def sensitivity(y_true, y_pred):
    """Return sensitivity (how many of the positives were detected?)"""
    tp = K.sum(true_positives(y_true, y_pred))
    fn = K.sum(false_negatives(y_true, y_pred))
    return tp / (tp + fn + K.epsilon())


def specificity(y_true, y_pred):
    """Return specificity (how many of the negatives were detected?)"""
    tn = K.sum(true_negatives(y_true, y_pred))
    fp = K.sum(false_positives(y_true, y_pred))
    return tn / (tn+fp + K.epsilon())