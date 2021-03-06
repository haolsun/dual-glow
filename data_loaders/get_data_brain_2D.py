import os
import tensorflow as tf
import numpy as np
import glob

_FILES_SHUFFLE = 1024
_SHUFFLE_FACTOR = 4

IMG_SIZE = 96

def parse_tfrecord_tf(record, res, rnd_crop):
    keys_to_features = {
        'id': tf.FixedLenFeature([], tf.string),
        'img_mri': tf.FixedLenFeature([IMG_SIZE * IMG_SIZE], dtype=tf.float32),
        'img_pet': tf.FixedLenFeature([IMG_SIZE * IMG_SIZE], dtype=tf.float32),

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

    img_mri = tf.reshape(features['img_mri'], [IMG_SIZE , IMG_SIZE])
    img_pet = tf.reshape(features['img_pet'], [IMG_SIZE , IMG_SIZE])

    img_mri = tf.expand_dims(img_mri, axis=2)
    img_pet = tf.expand_dims(img_pet, axis=2)

    return img_mri, img_pet, tf.cast(features['age'], tf.float32)


def input_fn(tfr_file, shards, rank, pmap, fmap, n_batch, resolution, rnd_crop, is_training):
    files = tf.data.Dataset.list_files(tfr_file)
    if ('lsun' not in tfr_file) or is_training:
        # For 'lsun' validation, only one shard and each machine goes over the full dataset
        # each worker works on a subset of the data
        files = files.shard(shards, rank)
    if is_training:
        # shuffle order of files in shard
        files = files.shuffle(buffer_size=_FILES_SHUFFLE)
    dset = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=fmap))
    if is_training:
        dset = dset.shuffle(buffer_size=n_batch * _SHUFFLE_FACTOR)
    dset = dset.repeat()
    dset = dset.map(lambda x: parse_tfrecord_tf(
        x, resolution, rnd_crop), num_parallel_calls=pmap)
    dset = dset.batch(n_batch)
    dset = dset.prefetch(1)
    itr = dset.make_one_shot_iterator()
    return itr


def get_tfr_file(data_dir, split):
    data_dir = os.path.join(data_dir, split)
    tfr_prefix = os.path.join(data_dir, os.path.basename(data_dir))
    # tfr_file = tfr_prefix + '-r%02d-s-*-of-*.tfrecords' % (res_lg2)  -00000-of-00060
    tfr_file = tfr_prefix + '*.tfrecords'  # + '-*-of-*'
    files = glob.glob(tfr_file)
    # assert len(files) == int(files[0].split(
    #     "-")[-1].split(".")[0]), "Not all tfrecords files present at %s" % tfr_prefix
    return tfr_file


def get_data(sess, data_dir, shards, rank, pmap, fmap, n_batch_train, n_batch_test, n_batch_init, resolution, rnd_crop):
    # assert resolution == 2 ** int(np.log2(resolution))

    train_file = get_tfr_file(data_dir, 'train')
    valid_file = get_tfr_file(data_dir, 'validation')

    train_itr = input_fn(train_file, shards, rank, pmap,
                         fmap, n_batch_train, resolution, rnd_crop, True)
    valid_itr = input_fn(valid_file, shards, rank, pmap,
                         fmap, n_batch_test, resolution, rnd_crop, False)

    data_init = make_batch(sess, train_itr, n_batch_train, n_batch_init)

    return train_itr, valid_itr, data_init


def make_batch(sess, itr, itr_batch_size, required_batch_size):
    ib, rb = itr_batch_size, required_batch_size
    # assert rb % ib == 0
    k = int(np.ceil(rb / ib))
    xs_mri, xs_pet, ys = [], [], []
    data = itr.get_next()
    for i in range(k):
        x_mri, x_pet, y = sess.run(data)
        xs_mri.append(x_mri)
        xs_pet.append(x_pet)
        ys.append(y)
    x_mri, x_pet, y = np.concatenate(xs_mri)[:rb], np.concatenate(xs_pet)[:rb], np.concatenate(ys)[:rb]
    return {'x_m': x_mri, 'x_p': x_pet, 'y': y}