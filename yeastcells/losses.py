import keras.backend as K
from keras.backend.common import epsilon
import tensorflow as tf


try:
    import tensorflow as tf
    installed_legacy_tf = tf.__version__ == '1.12.0'
except ImportError:
    installed_legacy_tf = False

float32 = tf.float32 if installed_legacy_tf else tf.dtypes.float32


def _to_tensor(x, dtype):
    return tf.convert_to_tensor(x, dtype=dtype)
  

def normalized_binary_crossentropy(target, output, from_logits=False):
    if not from_logits:
        _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.math.log(output / (1 - output))
    
    w0 = tf.dtypes.cast(K.sum(target), float32)
    s = K.shape(target)
    w1 = tf.dtypes.cast(s[0] * s[1] * s[2] * s[3], float32) - w0
    r = K.sqrt(w0*w0 + w1*w1)
    w0, w1 = w0 / r, w1 / r
    target2 = tf.dtypes.cast(target, float32)
    w = w0 * (1. - target2) + w1 * target2
    
    r = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=target, logits=output)
    return w * r


def auto_weighting_binary_crossentropy(target, output, from_logits=False):
    if not from_logits:
        _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.math.log(output / (1 - output))
    
    subsample = tf.dtypes.cast(tf.math.logical_or(
      tf.less(K.random_uniform(K.shape(target), minval=0, maxval=1), 0.02),
      tf.greater(target, 0.5)
    ), float32)
    
    r = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=target, logits=output)
    return r * subsample / 0.04


# def masked_loss(loss_function, ignore_value=2):
#   def f(target, output):
#     return (target != ignore_value) * loss_function(target, output)
#   return f

# def masked_accuracy(ignore_value=2):
#   def f(target, output):
# #     return (target != ignore_value) *  / (
# #     return (K.sum(
# #         K.cast(tf.Print(target != ignore_value, [target != ignore_value], "mask"), tf.float32)))
#     mask = tf.cast(tf.not_equal(target, ignore_value), tf.float32)
#     masked = mask * tf.cast(tf.equal(target, tf.cast(output > 0.5, tf.float32)), tf.float32)
    
#     return K.sum(masked) / K.sum(mask)
#   return f


#   import tensorflow as tf
#   from keras.backend.common import epsilon

#   def _to_tensor(x, dtype):
#       return tf.convert_to_tensor(x, dtype=dtype)

#   # copied from https://github.com/keras-team/keras/blob/master/keras/backend/tensorflow_backend.py#L3626

#   def auto_weighting_binary_crossentropy(target, output, from_logits=False):
#       # target and output are image segmentations of shape
#       # n_samples x width x height x 1

#       if not from_logits:
#           _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
#           output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
#           output = tf.math.log(output / (1 - output))

#       # subsample creates a mask (all elements are 0 or 1) of shape
#       # n_samples x width x height x 1. For the mask, 2% of the values
#       # where target is 0 is selected, and all values where target is 1
#       # are selected. This is because in my case 2% of the pixels belonged
#       # to the positive class.
#       subsample = tf.cast(tf.math.logical_or(
#         K.random_uniform(K.shape(target), minval=0, maxval=1) < 0.02,
#         target > 0.5
#       ), tf.float32)

#       r = tf.nn.sigmoid_cross_entropy_with_logits(
#           labels=target, logits=output)

#       # By multiplying the cross entropy with the subsample, a lot of the pixels
#       # are discarded, but there is (approximately) an equal amount of 1 pixels
#       # and 0 pixels.
#       return r * subsample
