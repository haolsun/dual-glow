import os
import tensorflow as tf
import numpy as np
import glob

_FILES_SHUFFLE = 1024
_SHUFFLE_FACTOR = 4


MRI_SHAPE = [48,64,48]
PET_SHAPE = [48,64,48]

def parse_tfrecord_tf(record, att_name):
    keys_to_features = {
        'id': tf.FixedLenFeature([], tf.string),
        'img_mri': tf.FixedLenFeature([np.prod(MRI_SHAPE)], dtype=tf.float32),
        'img_pet': tf.FixedLenFeature([np.prod(PET_SHAPE)], dtype=tf.float32),
        # 'shape_mri': tf.FixedLenFeature(
        #     [3], dtype=tf.int64, default_value=[121,145,121]),
        # 'shape_pet': tf.FixedLenFeature(
        #     [3], dtype=tf.int64, default_value=[79,95,79]),

        #'adas': tf.FixedLenFeature([], dtype=tf.float32),
        'age': tf.FixedLenFeature([], dtype=tf.float32),
        #'cdr': tf.FixedLenFeature([], dtype=tf.float32),
        'group': tf.FixedLenFeature([], dtype=tf.float32),
        #'apoe': tf.FixedLenFeature(
         #   [], dtype=tf.int64, default_value=-1),
        #'dxbl': tf.FixedLenFeature(
        #    [], dtype=tf.int64, default_value=-1),
        'gender': tf.FixedLenFeature(
            [], dtype=tf.int64, default_value=-1),
        #'mmse': tf.FixedLenFeature(
        #    [], dtype=tf.int64, default_value=-1),
        #'ravlt': tf.FixedLenFeature(
        #    [], dtype=tf.int64, default_value=-1),
    }

    features = tf.parse_single_example(record, features=keys_to_features)

    img_mri = tf.reshape(features['img_mri'], MRI_SHAPE)
    img_pet = tf.reshape(features['img_pet'], PET_SHAPE)

    img_mri = tf.expand_dims(img_mri, axis=3)
    img_pet = tf.expand_dims(img_pet, axis=3)

    att = features[att_name]

    return img_mri, img_pet,  tf.cast(att, tf.float32) # att



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
    dset = dset.map(lambda x: parse_tfrecord_tf(x, att), num_parallel_calls=pmap)
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


def get_data(sess, data_dir, shards, rank, pmap, fmap, n_batch_train, n_batch_test, n_batch_init, att):
    # assert resolution == 2 ** int(np.log2(resolution))

    train_file = get_tfr_file(data_dir, 'train')
    valid_file = get_tfr_file(data_dir, 'validation')

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
    return {'x_mri': x_mri, 'x_pet': x_pet, 'y': y}

