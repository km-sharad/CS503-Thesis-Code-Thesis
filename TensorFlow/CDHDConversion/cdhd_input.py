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

  for x in xrange(batch_size):
    anno_file_batch_rows.append(anno_file_lines[x])

  #images = np.array([])
  images = []
  meta_dict = {}
  #for image_idx in xrange(batch_size):
  for image_idx in xrange(5):
    #read and convert images into numpy array
    split_row = anno_file_batch_rows[image_idx].split('|')

    im = Image.open(FLAGS.data_dir + split_row[2])
    print im.size

    in_data = np.transpose(np.asarray(im, dtype=np.uint8),(1,0,2))
    print in_data.shape

    tensor_images = tf.convert_to_tensor(in_data)
    print tensor_images

    #img_numpy_arr = array(Image.open(FLAGS.data_dir + split_row[2]))
    #images.append(images, array(Image.open(FLAGS.data_dir + split_row[2])))
    
    #tensor_images = tf.convert_to_tensor(images)

    #print('*** len: ', len(images[0]), len(images[0][0]), len(images[0][0][0]))

    '''
    img_filenameQ = tf.train.string_input_producer([FLAGS.data_dir + split_row[2]],num_epochs=None)
    recordReader = tf.TFRecordReader()
    key, value = recordReader.read(img_filenameQ)

    #img_3d = tf.image.decode_jpeg(value, channels=3)
    img_3d = tf.decode_raw(value, tf.uint8)
    #print('*** ', img_3d)    
    '''

    meta_dict['gt_x_coord'] = split_row[0]
    meta_dict['gt_y_coord'] = split_row[1]
    meta_dict['img_size'] = [int(i) for i in split_row[3].split(',')]
    meta_dict['bbox'] = [int(i) for i in split_row[8][0:len(split_row[8]) - 2].split(',')]

  return images, meta_dict
