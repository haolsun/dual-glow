import os
import tensorflow as tf
import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt

_FILES_SHUFFLE = 1024
_SHUFFLE_FACTOR = 50

img_shape = [48, 64, 48]

def parse_fn(aslraw_path, fdgpet_path, att):
    def _load_npy(path):
        return np.load(path)
    asl_data = tf.py_func(_load_npy, [aslraw_path], tf.float32)
    pet_data = tf.py_func(_load_npy, [fdgpet_path], tf.float32)

    asl_data = tf.reshape(asl_data, img_shape)
    pet_data = tf.reshape(pet_data, img_shape)

    asl_data = tf.expand_dims(asl_data, axis=-1)
    pet_data = tf.expand_dims(pet_data, axis=-1)

    return asl_data, pet_data, att

def input_fn(meta_data, shards, rank, pmap, fmap, n_batch, is_training):
    dset = tf.data.Dataset.from_tensor_slices((meta_data['aslraw_path'].values, meta_data['fdgpet_path'].values, meta_data['att']))
    if is_training:
        dset = dset.shuffle(buffer_size=n_batch * _SHUFFLE_FACTOR)
    dset = dset.repeat()
    dset = dset.map(parse_fn, num_parallel_calls=pmap)
    dset = dset.batch(n_batch)
    dset = dset.prefetch(1)
    itr = dset.make_one_shot_iterator()
    return itr


def get_data(sess, data_dir, shards, rank, pmap, fmap, n_batch_train, n_batch_test, n_batch_init, att, fold):
    meta_data = pd.read_csv(os.path.join(data_dir, 'metadata.csv'))
    meta_data = meta_data.reset_index(drop=True)

    unique_individuals = sorted(meta_data['subject_id'].unique())

    if fold < len(unique_individuals):
        exclude = unique_individuals[fold]
        print("Excluding individual %s" % unique_individuals[fold])
    else:
        print("Individual index %d exceeds total number of individuals" % fold)

    new_meta = pd.DataFrame()
    new_meta['subject_id'] = meta_data['subject_id']
    new_meta['aslraw_path'] = meta_data['ASL_path']
    new_meta['fdgpet_path'] = meta_data['PET_path']
    new_meta['att'] = meta_data[att].values.astype(np.float32)

    train_meta = new_meta[new_meta['subject_id'] != exclude]
    valid_meta = new_meta[new_meta['subject_id'] == exclude]

    train_size = len(train_meta)
    valid_size = len(valid_meta)

    train_itr = input_fn(train_meta, shards, rank, pmap,
                         fmap, n_batch_train, True)
    valid_itr = input_fn(valid_meta, shards, rank, pmap,
                         fmap, n_batch_test,  False)

    data_init = make_batch(sess, train_itr, n_batch_train, n_batch_init)

    return train_itr, valid_itr, data_init, train_size, valid_size


def make_batch(sess, itr, itr_batch_size, required_batch_size):
    ib, rb = itr_batch_size, required_batch_size
    # assert rb % ib == 0
    k = int(np.ceil(rb / ib))
    xs_mri, xs_pet, ys = [], [], []
    data = itr.get_next()
    for _ in range(k):
        x_mri, x_pet, y = sess.run(data)
        xs_mri.append(x_mri)
        xs_pet.append(x_pet)
        ys.append(y)
    x_mri, x_pet, y = np.concatenate(xs_mri)[:rb], np.concatenate(xs_pet)[:rb], np.concatenate(ys)[:rb]
    return {'x_mri': x_mri, 'x_pet': x_pet, 'y': y}

def test():
    with tf.Session() as sess:
        train_it, _, _, train_size, test_size = get_data(sess, './data_loaders/asl2pet_unique', 1, 1, 1, 1, 4, 4, 4, 'year_diff', 0)
        img_asl, img_pet, y = sess.run(train_it.get_next())
        print(y)
        print(train_size, test_size)

if __name__ == '__main__':
    test()