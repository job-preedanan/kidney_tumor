import tensorflow as tf


# Recall
def recall(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred) + 1
    denominator = tf.reduce_sum(y_true) + 1
    return intersection / denominator


# Precision
def precision(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred) + 1
    denominator = tf.reduce_sum(y_pred) + 1
    return intersection / denominator


def dice_coef(y_true, y_pred):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred) + 1
    denominator = tf.reduce_sum(y_true ** 2) + tf.reduce_sum(y_pred ** 2) + 1
    return 2.0 * intersection / denominator


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)


def focal_tv_loss(beta, gamma):
    def loss(y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true * y_pred) + \
                      (1 - beta) * tf.reduce_sum((1 - y_true) * y_pred) + \
                      beta * tf.reduce_sum(y_true * (1 - y_pred))
        loss_value = (intersection + 1) / (denominator + 1)

        return tf.pow(1.0 - loss_value, (1/gamma))

    return loss