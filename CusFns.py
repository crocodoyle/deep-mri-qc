from keras import backend as K

def binary_sensitivity(y_true, y_pred):
    import tensorflow as tf
    y_true = tf.minimum(tf.maximum(y_true - 0.499, 0) * 10000, 1)
    y_pred = tf.minimum(tf.maximum(y_pred - 0.499, 0) * 10000, 1)

    cp = K.sum(y_true)
    tp_tensor = y_true * y_pred
    tp = K.sum(tp_tensor)
    return tp/cp

def binary_sensitivity_loss(y_true, y_pred):
    import tensorflow as tf
    y_true = tf.minimum(tf.maximum(y_true - 0.499, 0) * 10000, 1)
    y_pred = tf.minimum(tf.maximum(y_pred - 0.499, 0) * 10000, 1)
    cp = K.sum(y_true)
    tp_tensor = y_true * y_pred
    tp = K.sum(tp_tensor)
    return 1 - tp / cp

def binary_specificity(y_true, y_pred):
    import tensorflow as tf
    y_true = tf.minimum(tf.maximum(y_true - 0.499, 0) * 10000, 1)
    y_pred = tf.minimum(tf.maximum(y_pred - 0.499, 0) * 10000, 1)
    ones_tensor = tf.ones_like(y_true)
    cn_tensor = ones_tensor - y_true
    cn = K.sum(cn_tensor)
    y_pred_n = ones_tensor - y_pred
    tn = K.sum(cn_tensor * y_pred_n)
    return tn/cn

def binary_specificity_loss(y_true, y_pred):
    import tensorflow as tf
    y_true = tf.minimum(tf.maximum(y_true - 0.499, 0) * 10000, 1)
    y_pred = tf.minimum(tf.maximum(y_pred - 0.499, 0) * 10000, 1)
    ones_tensor = tf.ones_like(y_true)
    cn_tensor = ones_tensor - y_true
    cn = K.sum(cn_tensor)
    y_pred_n = ones_tensor - y_pred
    tn = K.sum(cn_tensor * y_pred_n)
    return 1 - tn/cn