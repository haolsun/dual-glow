import SimpleITK as sitk
import os, sys
import numpy as np
import random
import tensorflow as tf
from PIL import Image
from scipy  import ndimage
import glob
import xmltodict
import xml.etree.ElementTree

tf.app.flags.DEFINE_string('output_directory', '/home/haoliang/Downloads/datasets/adni_fdg/3D/',
                           'Output data directory')

tf.app.flags.DEFINE_string('pets_folder', '/home/haoliang/Downloads/datasets/adni_fdg/preprocessed/matched_pet_nii',
                           'Output data directory')

tf.app.flags.DEFINE_string('mris_folder', '/home/haoliang/Downloads/datasets/adni_fdg/preprocessed/matched_mri_nii',
                           'Output data directory')

tf.app.flags.DEFINE_string('xmls_pet', '/home/haoliang/Downloads/datasets/adni_fdg/pet_xml',
                           'xmls_pet directory')

tf.app.flags.DEFINE_string('xmls_mri', '/home/haoliang/Downloads/datasets/adni_fdg/mri_xml',
                           'xmls_mri directory')

tf.app.flags.DEFINE_string('mri_type', 'w',
                           'mri type: mwc1     w')


tf.app.flags.DEFINE_integer('side_info_type', 1,
                            '0 or 1;      0: mri, 1: pet.')

FLAGS = tf.app.flags.FLAGS




def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



def normalise_zero_one(image):
    """Image normalisation. Normalises image to fit [0, 1] range."""

    image = image.astype(np.float32)

    # minimum = np.min(image)
    # maximum = np.max(image)

    maximum, minimum = np.percentile(image, [99, 1])

    if maximum > minimum:
        ret = (image - minimum) / (maximum - minimum)
    else:
        ret = image * 0.
    return np.clip(ret, 0, 1.0)

def normalise_one_one(image):
    """Image normalisation. Normalises image to fit [-1, 1] range."""

    ret = normalise_zero_one(image)
    # ret *= 2.
    # ret -= 0.5
    return ret



# pets_folder = "/home/haoliang/Downloads/datasets/adni_new/preprocessed/matched_pet_nii"
# mris_folder  = "/home/haoliang/Downloads/datasets/adni_new/preprocessed/matched_mri_nii"

# mris = os.listdir(mris_folder)
# pets = os.listdir(pets_folder)



def imge_processing(_folder, _id,prefix, _crop, output_dim):
    _path = os.path.join(_folder, _id, '{}.nii'.format(prefix + _id))

    nii = sitk.GetArrayFromImage(sitk.ReadImage(str(_path)))

    # get rid of nan
    nii = np.nan_to_num(nii)

    # cropping
    nii = nii[_crop[0]:_crop[1], _crop[2]:_crop[3], _crop[4]:_crop[5]]



    # resize
    if output_dim[0] != nii.shape[0] or output_dim[1] != nii.shape[1] or output_dim[2] != nii.shape[2]:
        nii = ndimage.zoom(nii, [output_dim[0] / nii.shape[0], output_dim[1] / nii.shape[1], output_dim[2] / nii.shape[2]])


    # normalizing to [0  1]
    nii = normalise_zero_one(nii)

    return nii

def parse_xmls(id_pair,side_info_type):

    mri_id = id_pair[0]
    pet_id = id_pair[1][1:-1]
    if side_info_type==0:
        while mri_id[-1] == 'A': # remove A
            mri_id = mri_id[:-1]

        mri_id = mri_id.split('_')
        xml_name = 'ADNI_{}_S_{}*{}_{}.xml'.format(mri_id[0], mri_id[2], mri_id[3], mri_id[4])
        xml_path = os.path.join(FLAGS.xmls_mri, xml_name)
        xml_path = glob.glob(xml_path)


    elif side_info_type==1:
        pet_ids = pet_id.split('_')
        xml_name = 'ADNI_{}_S_{}*{}_{}.xml'.format(pet_ids[0], pet_ids[2], pet_ids[3], pet_ids[4])
        xml_path = os.path.join(FLAGS.xmls_pet, xml_name)
        xml_path = glob.glob(xml_path)
    else:
        sys.exit("side_info_type errors!")

    if len(xml_path) == 1:
        with open(xml_path[0]) as fd:
            doc_ = xmltodict.parse(fd.read())

        # as_list = doc_['idaxs']['project']['subject']['visit']['assessment']
        # as_dict = {}
        # for i in as_list:
        #     as_dict[i['@name']] = i['component']['assessmentScore']['#text']
        if doc_['idaxs']['project']['subject']['subjectSex'] == 'F':
            gender = 1
        else:
            gender = 0

        reor_group = {'CN': 0.0,
                       'SMC': 0.2,
                      'EMCI':0.4,
                      'MCI':0.6,
                      'LMCI':0.8,
                      'AD':1.0,}

        dict = {'age':    float(doc_['idaxs']['project']['subject']['study']['subjectAge'])/100,
                'gender': gender,
                'group': reor_group[doc_['idaxs']['project']['subject']['researchGroup']],
                # 'cdr':    doc_['idaxs']['project']['subject']['visit']['assessment'],
        # 'gds':,
        # 'mmse':,
        # 'faq':
        }
        # print()
    else:
        sys.exit("No xml file found!")

    return dict

def _process_dataset(name, data_list, shuffle=True, side_info_type=0):
    if shuffle:
        random_order = random.sample(range(len(data_list)), len(data_list))
    else:
        random_order = range(len(data_list))

    n_volume_per_shard = 10
    for i, value in enumerate(random_order):
        item = data_list[value]
        id_pair = item.split(',')

        mri_id = id_pair[0]
        pet_id = id_pair[1][1:-1]

        # dict = parse_xmls(id_pair, side_info_type)

        if FLAGS.mri_type == 'mwc1':
            crop = [8, 104, 10, 138, 11, 107]
        elif FLAGS.mri_type == 'w':
            crop = [8, 104, 10, 138, 4, 100]

        mri_nii = imge_processing(FLAGS.mris_folder,  mri_id, FLAGS.mri_type,
                                  crop, [48, 64, 48])#[65, 128, 128])
        pet_nii = imge_processing(FLAGS.pets_folder, pet_id, 'wr',
                                  [8, 104, 0, 127, 4, 100], [48, 64, 48])


        # img = Image.fromarray(((mri_nii[48, :, :] + 0.5)*255).astype(np.uint8))
        # # img.show()
        # img.save('./check/' + mri_id + '.png')
        #
        # img2 = Image.fromarray(((pet_nii[48, :, :] + 0.5) * 255).astype(np.uint8))
        # # img2.show()
        # img2.save('./check/' + pet_id + '.png')

        if not (i) % n_volume_per_shard:
            output_file = os.path.join(FLAGS.output_directory, name,
                                       name + str(int((i) / n_volume_per_shard)) + '.tfrecords')
            train_filename = output_file  # address to save the TFRecords file

            # open the TFRecords file
            writer = tf.python_io.TFRecordWriter(train_filename)
            print(name + ' data: {}/{}'.format((i), len(data_list)))
            sys.stdout.flush()

        feature = {'id': _bytes_feature(bytes(mri_id[:10], 'utf-8')),
                   'img_mri': _float_feature(list(mri_nii.flatten())),
                   'img_pet': _float_feature(list(pet_nii.flatten())),
                   # 'shape_mri':_int64_feature(list(mri_list[i].shape)),
                   # 'shape_pet':_int64_feature(list(pet_list[i].shape)),
                   # 'adas': _float_feature(sideinfo['adas'].item(i)),
                   'age': _float_feature(dict['age']),
                   # 'cdr': _float_feature(sideinfo['cdr'].item(i)),
                   # 'apoe': _int64_feature(sideinfo['apoe'].item(i)),
                   # 'dxbl': _int64_feature(sideinfo['dxbl'].item(i)),
                   'gender': _int64_feature(dict['gender']),
                   'group': _float_feature(dict['group'])
                   # 'mmse': _int64_feature(sideinfo['mmse'].item(i)),
                   # 'ravlt': _int64_feature(sideinfo['ravlt'].item(i))
                   }

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

        # size = mri_nii.shape[0]
        # if shuffle:
        #     j_list = random.sample(range(size), size)
        # else:
        #     j_list = range(size)
        # for j in j_list:
        #     # for j in range(mri_list[i].shape[0]):
        #     # kkk=list(mri_list[i][:, :, j].flatten())
        #     # Create a feature


        if not ((i) + 1) % n_volume_per_shard:
            writer.close()

    sys.stdout.flush()

    print()





def main(unused_argv):
  #
  # n_dataset = 690
  # partition_ratio = 0.9
  print('MRI image type: ' + FLAGS.mri_type)
  with open('./preprocessed/txt/goodlist.txt', 'r') as f:
      data_list = list(f)


  _process_dataset('train', data_list[:709], shuffle=True, side_info_type=FLAGS.side_info_type)
  _process_dataset('validation', data_list[709:], shuffle=False, side_info_type=FLAGS.side_info_type)

  # _process_dataset('validation', [int(partition_ratio * n_dataset), n_dataset],
  #                  FLAGS.output_directory, id_list, mri_list, pet_list, sideinfo, shuffle=False)

if __name__ == '__main__':
  tf.app.run()


# t1 = t1[..., np.newaxis]


# t1 = t1[len(t1) // 2 - 5:len(t1) // 2 + 5]

# t1 = normalise_one_one(t1)
#
# images = t1
#
# print(images.shape)
