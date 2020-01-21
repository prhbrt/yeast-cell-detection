import keras.backend as K
import tensorflow as tf


def precision(target, output):
  P = output > 0.5
  TP = tf.logical_and(P, target > 0)
  return K.sum(tf.cast(TP, tf.float32)) / K.sum(tf.cast(P, tf.float32))


def recall(target, output):
  P = target > 0
  TP = tf.logical_and(P, output > 0.5)
  return K.sum(tf.cast(TP, tf.float32)) / K.sum(tf.cast(P, tf.float32))


def k_min(y_true, y_pred):
  return K.min(y_pred)


def k_max(y_true, y_pred):
  return K.max(y_pred)


def k_mean(y_true, y_pred):
  return K.mean(y_pred > 0.5)
