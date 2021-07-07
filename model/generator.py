import tensorflow as tf

import os
import time

from matplotlib import pyplot as plt
from IPython import display

from model_utils import gendisc_upunit, gendisc_downunit

def Generator():
    inputs = tf.keras.layers.Input([256, 256, 3])

    down_stack = [
                  
    gendisc_downunit(64, 4, apply_batchnorm=False),
    gendisc_downunit(128, 4),
    gendisc_downunit(256, 4),
    gendisc_downunit(512, 4),
    gendisc_downunit(512, 4),
    gendisc_downunit(512, 4),
    gendisc_downunit(512, 4),
    gendisc_downunit(512, 4),

    ]

    up_stack = [
               
    gendisc_upunit(512, 4, apply_dropout=True),
    gendisc_upunit(512, 4, apply_dropout=True),
    gendisc_upunit(512, 4, apply_dropout=True),
    gendisc_upunit(512, 4),
    gendisc_upunit(256, 4),
    gendisc_upunit(128, 4),
    gendisc_upunit(64, 4),

    ]

    init = tf.random_uniform_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding="SAME", kernel_initializer=init, activation='tanh')

    x = inputs

    skips = []

    for downsample in down_stack:
        x = downsample(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)

    return model

generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

LAMBDA = 100

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss
