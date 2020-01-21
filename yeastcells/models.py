from keras.models import Model, load_model
from keras.layers import *

from .losses import normalized_binary_crossentropy
from .metrics import precision, recall, k_min, k_max, k_mean


class UNET(Model):
    def __init__(self):
        """Copied from https://github.com/zhixuhao/unet/blob/master/model.py"""
        inputs = Input([None, None, 1])

        nfilters = 16 # in the original architecture, this was 64

        conv1 = Conv2D(nfilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(nfilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        nfilters *= 2
        last_out = pool1

        conv2 = Conv2D(nfilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(last_out)
        conv2 = Conv2D(nfilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        nfilters *= 2
        last_out = pool2

        conv3 = Conv2D(nfilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(last_out)
        conv3 = Conv2D(nfilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        nfilters *= 2
        last_out = pool3

        conv4 = Conv2D(nfilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(last_out)
        conv4 = Conv2D(nfilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(rate=0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
        nfilters *= 2
        last_out = pool4

        conv5 = Conv2D(nfilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(last_out)
        conv5 = Conv2D(nfilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(rate=0.5)(conv5)
        nfilters //= 2 # // is for integer division, where / is floating point division in Python 3
        last_out = drop5

        up6 = Conv2D(nfilters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(last_out))
        merge6 = concatenate([drop4,up6], axis = 3)

        conv6 = Conv2D(nfilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(nfilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
        nfilters //= 2
        last_out = conv6

        up7 = Conv2D(nfilters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(last_out))
        merge7 = concatenate([conv3,up7], axis = 3)
        conv7 = Conv2D(nfilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(nfilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        nfilters //= 2
        last_out = conv7
        
        up8 = Conv2D(nfilters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(last_out))
        merge8 = concatenate([conv2,up8], axis = 3)
        conv8 = Conv2D(nfilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(nfilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        nfilters //= 2
        last_out = conv8

        up9 = Conv2D(nfilters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(last_out))
        merge9 = concatenate([conv1,up9], axis = 3)

        conv9 = Conv2D(nfilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(nfilters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

        super(UNET, self).__init__(inputs=inputs, outputs=conv10)


def load_unet_model(filename):
    return load_model(filename, custom_objects={
        'normalized_binary_crossentropy': normalized_binary_crossentropy,
        'precision': precision,
        'recall': recall,
        'k_min': k_min,
        'k_max': k_max,
        'k_mean': k_mean,
        'UNET': UNET
    })
