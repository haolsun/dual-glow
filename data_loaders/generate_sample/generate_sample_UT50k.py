from PIL import Image
import tensorflow as tf
import numpy as np
import sys
sys.path.append("..")
from get_data_UT50k import *
from config import ModelConfig

def main():
    # img, label = parse_tfrecord_tf(record='/home/haoliang/Downloads/datasets/imagenet-full/validation/validation-00000-of-00060',
    #                   res=64, rnd_crop = True)

    tf_file_name = '/home/ericaragorn/VikasLab/dual-glow/dual-glow-ycond/data_loaders/UT50k'
    att = ModelConfig.attributes['UT50k']

    sess = tf.Session()



    # train_iterator, test_iterator, data_init = \
    #     get_data(sess, tf_file_name, 2, 1, 16, 1, 64,
    #                32, 64, False)
    # sess.run(tf.global_variables_initializer())
    train_iterator, test_iterator ,_= \
             get_data(sess, tf_file_name,2, 1, 1, 1, 96,
                      96, 96, att)

    # sess.run(tf.global_variables_initializer())

    input_, target_, ys= test_iterator.get_next()


    IMG_m, IMG_p, y = sess.run([input_, target_, ys])

    print()

    # img = Image.fromarray((IMG_m[1][:,:,0]  )*255)
    # img.show()
    # img = Image.fromarray((IMG_m[10][:,:,0] )*255)
    # img.show()
    # img = Image.fromarray((IMG_m[20][:,:,0] )*255)
    # img.show()
    # img = Image.fromarray((IMG_m[30][:,:,0] )*255)
    # img.show()

    img = Image.fromarray((IMG_p[0][:,:,0]   ) *255)
    img.show()

    img = Image.fromarray((IMG_p[5][:, :, 0]) * 255)
    img.show()

    img = Image.fromarray((IMG_p[10][:, :, 0]) * 255)
    img.show()

    img = Image.fromarray((IMG_p[15][:, :, 0]) * 255)
    img.show()

    img = Image.fromarray((IMG_p[20][:, :, 0]) * 255)
    img.show()

    img = Image.fromarray((IMG_p[25][:, :, 0]) * 255)
    img.show()

    img = Image.fromarray((IMG_p[30][:, :, 0]) * 255)
    img.show()

    img = Image.fromarray((IMG_p[35][:, :, 0]) * 255)
    img.show()

    img = Image.fromarray((IMG_p[40][:, :, 0]) * 255)
    img.show()

    img = Image.fromarray((IMG_p[45][:, :, 0]) * 255)
    img.show()

    img = Image.fromarray((IMG_p[49][:, :, 0]) * 255)
    img.show()
    # img = Image.fromarray((IMG_p[10][:,:,0]   ) *255)
    # img.show()
    # img = Image.fromarray((IMG_p[20][:,:,0]   ) *255)
    # img.show()
    # img = Image.fromarray((IMG_p[30][:,:,0]   ) *255)
    # img.show()


    np.save('../UT50k_sample_m', IMG_m)
    np.save('../UT50k_sample_p', IMG_p)
    np.save('../UT50k_sample_labels', y)


if __name__ == "__main__":
    main()