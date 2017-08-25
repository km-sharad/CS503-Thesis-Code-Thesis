import tensorflow as tf
import random
import numpy as np
from PIL import Image
from numpy import array

FLAGS = tf.app.flags.FLAGS

#
#tf.app.flags.DEFINE_string('data_dir', '/home/sharad/CS503-Thesis/car_dataset/', """Path to the CDHD data directory.""")

#local
tf.app.flags.DEFINE_string('data_dir', '../../../../car_dataset/', """Path to the CDHD data directory.""")
tf.app.flags.DEFINE_integer('total_visible_training_images', 1920,
                              """Number of training images where car door handle is visible.""")
tf.app.flags.DEFINE_string('max_im_side', 500, """Default max image side.""")


def distorted_inputs(batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """

  all_train_visible_idx = [x for x in xrange(0,FLAGS.total_visible_training_images)]
  random.shuffle(all_train_visible_idx)
  batch_indexes = all_train_visible_idx[0:batch_size]

  anno_file_batch_rows = []
  anno_file = open('cdhd_anno_training_data.txt')
  anno_file_lines = anno_file.readlines()

  for x in batch_indexes:
    anno_file_batch_rows.append(anno_file_lines[x])

  #images = np.array([])
  images = []
  meta_dict = {}
  #for image_idx in xrange(batch_size):
  for image_idx in xrange(5):
    #read and convert images into numpy array
    meta_rec = anno_file_batch_rows[image_idx].split('|')
    getImage(meta_rec)

  return images, meta_dict

def getImage(meta_rec):
    im_meta_dict = {}
    im = Image.open(FLAGS.data_dir + meta_rec[2])
    print meta_rec[2]
    print im.size

    im_meta_dict['gt_x_coord'] = meta_rec[0]
    im_meta_dict['gt_y_coord'] = meta_rec[1]
    im_meta_dict['img_size'] = [int(i) for i in meta_rec[3].split(',')]
    im_meta_dict['bbox'] = [int(i) for i in meta_rec[8][0:len(meta_rec[8]) - 2].split(',')]    

    im_np_arr = np.transpose(np.asarray(im, dtype=np.uint8),(1,0,2))
    print im_np_arr.shape

    getAugmentedImage(im_np_arr, im_meta_dict)

    #Compute the default scaling
    scale = round(FLAGS.max_im_side/float(np.amax(im_np_arr.shape)),2)

    #tensor_images = tf.convert_to_tensor(im_np_arr)
    #print tensor_images

def getAugmentedImage(im, im_meta_dict):
  lrFlipCDHDDataRecord(im, im_meta_dict)  

def lrFlipCDHDDataRecord(im, im_meta_dict):
  if random.uniform(0, 1) > 0.5:
    im = np.fliplr(im)
    print('after flip: ', im.shape)
    return im, im_meta_dict
  else:
    return im, im_meta_dict


