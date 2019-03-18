import os
import tensorflow as tf
import numpy as np
import glob
from config import ModelConfig

_FILES_SHUFFLE = 1024
_SHUFFLE_FACTOR = 4

INPUT_DIM = ModelConfig.input_dim['UT50k']
OUTPUT_DIM = ModelConfig.output_dim['UT50k']
## adaptive resolution


def parse_tfrecord_tf(record, att_names):

    keys_to_features = {
        'cid': tf.FixedLenFeature([], tf.string),
        'image/raw': tf.FixedLenFeature([], tf.string),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/channel': tf.FixedLenFeature([], tf.int64),
        'edge/raw': tf.FixedLenFeature([], tf.string),
        'edge/height': tf.FixedLenFeature([], tf.int64),
        'edge/width': tf.FixedLenFeature([], tf.int64),
        'bin_class': tf.FixedLenFeature([4], tf.int64),
        'bin_gender': tf.FixedLenFeature([4], tf.int64),
        'bin_material': tf.FixedLenFeature([5], tf.int64)
    }

    features = tf.parse_single_example(record, features=keys_to_features)

    input_img = tf.decode_raw(features['edge/raw'], out_type=tf.float32)
    target_img = tf.decode_raw(features['image/raw'], out_type=tf.float32)
    # input_ = tf.reshape(input_img, tf.stack([features['image/height'], features['image/width'], features['image/channel']], axis=0))
    # target_ = tf.reshape(target_img, tf.stack([features['edge/height'], features['edge/width']], axis=0))
    input_ = tf.reshape(input_img, INPUT_DIM)
    target_ = tf.reshape(target_img, OUTPUT_DIM)

    # img_mri = tf.expand_dims(img, axis=3)
    # target_ = tf.expand_dims(target_, axis=2)

    att_list = []
    for att_n in att_names:
        att_list.append(features[att_n])
    att = tf.concat(att_list, axis=0)

    return input_, target_, tf.cast(att, tf.float32)


def input_fn(tfr_file, shards, rank, pmap, fmap, n_batch, is_training, att):
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
        x, att), num_parallel_calls=pmap)
    dset = dset.batch(n_batch)
    dset = dset.prefetch(1)
    itr = dset.make_one_shot_iterator()
    return itr


def get_tfr_file(data_dir, split, prefix=None):
    data_dir = os.path.join(data_dir, split)
    if prefix is None:
        prefix = os.path.basename(data_dir)
    tfr_prefix = os.path.join(data_dir, prefix)
    # tfr_file = tfr_prefix + '-r%02d-s-*-of-*.tfrecords' % (res_lg2)  -00000-of-00060
    tfr_file = tfr_prefix + '*.tfrecord'  # + '-*-of-*'
    # files = glob.glob(tfr_file)
    # assert len(files) == int(files[0].split(
    #     "-")[-1].split(".")[0]), "Not all tfrecords files present at %s" % tfr_prefix
    return tfr_file


def get_data(sess, data_dir, shards, rank, pmap, fmap, n_batch_train, n_batch_test, n_batch_init, att):
    # assert resolution == 2 ** int(np.log2(resolution))

    train_file = get_tfr_file(data_dir, 'train', 'UT50k')
    valid_file = get_tfr_file(data_dir, 'test', 'UT50k')

    train_itr = input_fn(train_file, shards, rank, pmap,
                         fmap, n_batch_train, True, att)
    valid_itr = input_fn(valid_file, shards, rank, pmap,
                         fmap, n_batch_test,  False, att)

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