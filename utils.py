import tensorflow as tf

import os
import time

from matplotlib import pyplot as plt
from IPython import display

#Constants
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_HEIGHT = 256
IMG_WIDTH = 256
OUTPUT_CHANNELS = 3
PATH = "/content/edges2shoes/"

#Util functions
def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  w = tf.shape(image)[1]

  w = w // 2
  src_image = image[:, :w, :]
  tar_image = image[:, w:, :]

  src_image = tf.cast(src_image, tf.float32)
  tar_image = tf.cast(tar_image, tf.float32)

  return src_image, tar_image

def resize(in_img, out_img, height, width):
  src_image = tf.image.resize(in_img, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  tar_image = tf.image.resize(out_img, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return src_image, tar_image

def random_crop(src_image, tar_image):
  stacked_image = tf.stack([src_image, tar_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]

def normalize(src_image, tar_image):
  src_image = (src_image / 127.5) - 1
  tar_image = (tar_image / 127.5) - 1

  return src_image, tar_image

#TF Helper Functions
@tf.function()
def random_jitter(src_image, tar_image):

  src_image, tar_image = resize(src_image, tar_image, 286, 286)
  src_image, tar_image = random_crop(src_image, tar_image)

  if tf.random.uniform(()) > 0.5:

    src_image = tf.image.flip_left_right(src_image)
    tar_image = tf.image.flip_left_right(tar_image)

  return src_image, tar_image

def load_image_train(image_file):
  src_image, tar_image = load(image_file)
  src_image, tar_image = random_jitter(src_image, tar_image)
  src_image, tar_image = normalize(src_image, tar_image)

  return src_image, tar_image

def load_image_test(image_file):
  src_image, tar_image = load(image_file)
  src_image, tar_image = resize(src_image, tar_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  src_image, tar_image = normalize(src_image, tar_image)

  return src_image, tar_image

