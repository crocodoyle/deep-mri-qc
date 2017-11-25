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
    return K.sum(K.argmax(y_true) * K.argmax(y_pred))

def true_negatives(y_true, y_pred):
    """Return number of true negatives"""
    return K.sum((1 - K.argmax(y_pred)) * (1 - K.argmax(y_true)))


def false_positives(y_true, y_pred):
    """Return number of false positives"""
    return K.sum((K.argmax(y_pred)) * (1 - K.argmax(y_true)))


def false_negatives(y_true, y_pred):
    """Return number of false negatives"""
    return K.sum((1 - K.argmax(y_pred)) * (K.argmax(y_true)))


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