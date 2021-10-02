from tensorflow.python.keras.layers import Add, Conv2D, Input, Lambda, Conv2DTranspose
from tensorflow.python.keras.models import Model

from model.common import normalize, denormalize, pixel_shuffle

import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
import time
import os

# Network hyper-parameters:
kernel_size  =  4
filters_orig = 32
layer_depth  =  4

scale_factor = 4

ResizeToSame = lambda name: Lambda(
    lambda images: tf.image.resize(images[0], tf.shape(images[1])[-3:-1]),
    # `images` is a tuple of 2 tensors.
    # We resize the first image tensor to the shape of the 2nd
    name=name)

Upscale = lambda name: Lambda(
    lambda images: tf.image.resize(images, tf.shape(images)[-3:-1] * scale_factor),
    name=name)


def dae(kernel_size=3, filters_orig=16, layer_depth=4):
    x_in = Input(shape=(None, None, 3))
    x = Lambda(normalize)(x_in)

    x = simple_dae_for_super_resolution(x, kernel_size, filters_orig, layer_depth)
    x = Lambda(denormalize)(x)
    return Model(x_in, x, name="DAE")


def simple_dae_for_super_resolution(inputs, kernel_size=3, filters_orig=16, layer_depth=4):
    resized_inputs = Upscale(name='upscale_input')(inputs)
    decoded = simple_dae(resized_inputs, kernel_size, filters_orig, layer_depth)
    decoded = ResizeToSame(name='dec_output_scale')([decoded, resized_inputs])
    return decoded




def simple_dae(inputs, kernel_size=3, filters_orig=16, layer_depth=4):
    # Encoding layers:
    filters = filters_orig  # 16
    x = inputs
    # for i in range(4):  >> 0, 1, 2, 3
    # same : if the size of the filter is k, padding is given by k/2 in all directions.
    for i in range(layer_depth):
        x = Conv2D(filters=filters, kernel_size=kernel_size,
                   activation='relu', strides=2, padding='same',
                   name='enc_conv{}'.format(i))(x)

        filters = min(filters * 2, 512)

    # Decoding layers:
    for i in range(layer_depth):
        filters = max(filters // 2, filters_orig)
        x = Conv2DTranspose(filters=filters, kernel_size=kernel_size,
                            activation='relu', strides=2, padding='same',
                            name='dec_deconv{}'.format(i))(x)

    decoded = Conv2D(filters=inputs.shape[-1], kernel_size=1,
                     activation='sigmoid', padding='same',
                     name='dec_output')(x)
    # note: we use a sigmoid for the last activation, as we want the output values
    # to be between 0 and 1, like the input ones.

    return decoded






