import tensorflow as tf

import os
import time

from matplotlib import pyplot as plt
from IPython import display

def gendisc_downunit(filters, size, apply_batchnorm=True):
    init = tf.random_uniform_initializer(0., 0.02)
    
    gen_block = tf.keras.Sequential()
    gen_block.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='SAME', kernel_initializer=init, use_bias=False))

    if apply_batchnorm:
        gen_block.add(tf.keras.layers.BatchNormalization())

    gen_block.add(tf.keras.layers.LeakyReLU())

    return gen_block
  
def gendisc_upunit(filters, size, apply_dropout=True):
    init = tf.random_uniform_initializer(0., 0.02)

    gen_block = tf.keras.Sequential()
    gen_block.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding="SAME",kernel_initializer=init, use_bias=False))
    gen_block.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        gen_block.add(tf.keras.layers.Dropout(0.5))

    gen_block.add(tf.keras.layers.ReLU())

    return gen_block
