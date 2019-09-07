import os
import tensorflow as tf
import numpy as np
import glob
from tensorflow.python import debug as tf_debug


def parse_tfrecord_tf(record):
    keys_to_features = {
        'id': tf.FixedLenFeature([], tf.string),
        'img_mri': tf.FixedLenFeature([121 * 145 * 121], dtype=tf.float32),
        'img_pet': tf.FixedLenFeature([79 * 95 * 79], dtype=tf.float32),
        'shape_mri': tf.FixedLenFeature(
            [3], dtype=tf.int64, default_value=[121,145,121]),
        'shape_pet': tf.FixedLenFeature(
            [3], dtype=tf.int64, default_value=[79,95,79]),

        'adas': tf.FixedLenFeature([], dtype=tf.float32),
        'age': tf.FixedLenFeature([], dtype=tf.float32),
        'cdr': tf.FixedLenFeature([], dtype=tf.float32),

        'apoe': tf.FixedLenFeature(
            [], dtype=tf.int64, default_value=-1),
        'dxbl': tf.FixedLenFeature(
            [], dtype=tf.int64, default_value=-1),
        'gender': tf.FixedLenFeature(
            [], dtype=tf.int64, default_value=-1),
        'mmse': tf.FixedLenFeature(
            [], dtype=tf.int64, default_value=-1),
        'ravlt': tf.FixedLenFeature(
            [], dtype=tf.int64, default_value=-1),
    }

    features = tf.parse_single_example(record, features=keys_to_features)

    img_mri = tf.reshape(features['img_mri'], [121,145,121])
    img_pet = tf.reshape(features['img_pet'], [79,95,79])
    att = features['age']

    imgs_mri, imgs_pet, atts = tf.train.shuffle_batch([img_mri, img_pet, att],
                                                 batch_size=2,
                                                 capacity=10,
                                                 num_threads=2,
                                                 min_after_dequeue=5)

    return imgs_mri, imgs_pet, atts



tfrecords_filename = '/home/haoliang/Downloads/datasets/Brain_img/validation/validation.tfrecords'

filename_queue = tf.train.string_input_producer(
    [tfrecords_filename], num_epochs=1)

reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)

img_mri, img_pet, sideinfo = parse_tfrecord_tf(serialized_example)

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())


with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    IMG_mri = sess.run(img_mri)

    coord.request_stop()
    coord.join(threads)
    print()
