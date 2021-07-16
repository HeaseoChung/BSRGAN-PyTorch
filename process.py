"""Forward processing of raw data to sRGB images.

Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def apply_gains(bayer_images, red_gains, blue_gains):
  """Applies white balance gains to a batch of Bayer images."""
  bayer_images.shape.assert_is_compatible_with((None, None, None, 4))
  green_gains = tf.ones_like(red_gains)
  gains = tf.stack([red_gains, green_gains, green_gains, blue_gains], axis=-1)
  gains = gains[:, tf.newaxis, tf.newaxis, :]
  return bayer_images * gains


def demosaic(bayer_images):
  """Bilinearly demosaics a batch of RGGB Bayer images."""
  bayer_images.shape.assert_is_compatible_with((None, None, None, 4))

  # This implementation exploits how edges are aligned when upsampling with
  # tf.image.resize_bilinear().

  with tf.name_scope('demosaic'):
    shape = tf.shape(bayer_images)
    shape = [shape[1] * 2, shape[2] * 2]

    red = bayer_images[Ellipsis, 0:1]
    red = tf.image.resize(red, shape, method=tf.image.ResizeMethod.BILINEAR)

    green_red = bayer_images[Ellipsis, 1:2]
    green_red = tf.image.flip_left_right(green_red)
    green_red = tf.image.resize(green_red, shape, method=tf.image.ResizeMethod.BILINEAR)
    green_red = tf.image.flip_left_right(green_red)
    green_red = tf.nn.space_to_depth(green_red, 2)

    green_blue = bayer_images[Ellipsis, 2:3]
    green_blue = tf.image.flip_up_down(green_blue)
    green_blue = tf.image.resize(green_blue, shape, method=tf.image.ResizeMethod.BILINEAR)
    green_blue = tf.image.flip_up_down(green_blue)
    green_blue = tf.nn.space_to_depth(green_blue, 2)

    green_at_red = (green_red[Ellipsis, 0] + green_blue[Ellipsis, 0]) / 2
    green_at_green_red = green_red[Ellipsis, 1]
    green_at_green_blue = green_blue[Ellipsis, 2]
    green_at_blue = (green_red[Ellipsis, 3] + green_blue[Ellipsis, 3]) / 2

    green_planes = [
        green_at_red, green_at_green_red, green_at_green_blue, green_at_blue
    ]
    green = tf.nn.depth_to_space(tf.stack(green_planes, axis=-1), 2)

    blue = bayer_images[Ellipsis, 3:4]
    blue = tf.image.flip_up_down(tf.image.flip_left_right(blue))
    blue = tf.image.resize(blue, shape, method=tf.image.ResizeMethod.BILINEAR)
    blue = tf.image.flip_up_down(tf.image.flip_left_right(blue))

    rgb_images = tf.concat([red, green, blue], axis=-1)
    return rgb_images


def apply_ccms(images, ccms):
  """Applies color correction matrices."""
  images.shape.assert_has_rank(4)
  images = images[:, :, :, tf.newaxis, :]
  ccms = ccms[:, tf.newaxis, tf.newaxis, :, :]
  return tf.reduce_sum(images * ccms, axis=-1)


def gamma_compression(images, gamma=2.2):
  """Converts from linear to gamma space."""
  # Clamps to prevent numerical instability of gradients near zero.
  return tf.maximum(images, 1e-8) ** (1.0 / gamma)


def smoothstep(image):
  """a global tone mapping curve."""
  image = tf.clip_by_value(image, 0.0, 1.0)
  return 3.0 * image * image - 2.0 * image * image * image


def process(bayer_images, red_gains, blue_gains, cam2rgbs):
  """Processes a batch of Bayer RGGB images into sRGB images."""
  bayer_images.shape.assert_is_compatible_with((None, None, None, 4))
  with tf.name_scope('process'):
    # White balance.
    bayer_images = apply_gains(bayer_images, red_gains, blue_gains)
    # Demosaic.
    bayer_images = tf.clip_by_value(bayer_images, 0.0, 1.0)
    images = demosaic(bayer_images)
    # Color correction.
    images = apply_ccms(images, cam2rgbs)
    # Gamma compression.
    images = tf.clip_by_value(images, 0.0, 1.0)
    images = gamma_compression(images)
    images = smoothstep(images)
  return images