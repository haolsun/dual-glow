import os
import tensorflow as tf
import numpy as np
import glob
from get_data_ASL2PET import get_data
import matplotlib.pyplot as plt
import nibabel as nib

_FILES_SHUFFLE = 1024
_SHUFFLE_FACTOR = 4

att = 'year_diff'

def main():
    tf_file_name = '/home/ericaragorn/VikasLab/dual-glow-3D/data_loaders/asl2pet_unique'
    sess = tf.Session()
    _, test_iterator, _, = \
            get_data(sess, tf_file_name, 2, 1, 1, 1,70, 1, 4, att)

    sess.run(tf.global_variables_initializer())

    imgs_mri, imgs_pet, ys= test_iterator.get_next()

    IMG_m, IMG_p, y = sess.run([imgs_mri, imgs_pet, ys])

    from PIL import Image
    img = Image.fromarray((IMG_m[0][:, :, 24, 0]) * 255)
    img.show()
    # img = Image.fromarray((IMG_m[1][:, :, 24, 0]) * 255)
    # img.show()
    # img = Image.fromarray((IMG_m[2][:, :, 24, 0]) * 255)
    # img.show()
    # img = Image.fromarray((IMG_m[3][:, :, 24, 0]) * 255)
    # img.show()


    img = Image.fromarray((IMG_p[0][:, :, 24, 0]) * 255)
    img.show()
    # img = Image.fromarray((IMG_p[1][:, :, 24, 0]) * 255)
    # img.show()
    # img = Image.fromarray((IMG_p[2][:, :, 24, 0]) * 255)
    # img.show()
    # img = Image.fromarray((IMG_p[3][:, :, 24, 0]) * 255)
    # img.show()

    print("ASL Range:[%d, %d]", IMG_m.min(), IMG_m.max())
    print("PET Range:[%d, %d]", IMG_p.min(), IMG_p.max())

    plt.hist(IMG_m.flatten())
    plt.show()

    np.save('asl2pet_sample_m', IMG_m)
    np.save('asl2pet_sample_p', IMG_p)

    # ################### change ########################
    np.save('asl2pet_sample_label_' + att, y)

    print()

    # for i in range(4):
    #     nii = nib.Nifti1Image(IMG_p[i,:,:,:], np.eye(4))
    #     nib.save(nii, 'PET_sub_{}_test2.nii'.format(i))


    # img = Image.fromarray(IMG[0], 'RGB')
    # img.show()
    # img = Image.fromarray(IMG[1], 'RGB')
    # img.show()
    # img = Image.fromarray(IMG[2], 'RGB')
    # img.show()
    # img = Image.fromarray(IMG[3], 'RGB')
    # img.show()
    #
    # np.save('./imagenet', IMG)

if __name__ == "__main__":
    main()
