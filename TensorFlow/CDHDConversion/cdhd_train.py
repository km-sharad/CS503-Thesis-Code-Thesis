import cdhd
import random
import tensorflow as tf
from PIL import Image
from math import sqrt
import numpy as np
import time
import cdhd_input
from tensorflow.python import debug as tf_debug

FLAGS = tf.app.flags.FLAGS

#server
#tf.app.flags.DEFINE_string('data_dir', '/home/sharad/CS503-Thesis/car_dataset/', """Path to the CDHD data directory.""")

#local
tf.app.flags.DEFINE_string('data_dir', '../../../../car_dataset/', """Path to the CDHD data directory.""")
tf.app.flags.DEFINE_integer('total_visible_training_images', 1920,
                              """Number of training images where car door handle is visible.""")
tf.app.flags.DEFINE_integer('max_steps', 26000,"""Number of batches to run.""")
tf.app.flags.DEFINE_integer('stats_sample_size', 200,"""Number of images to calculate mean and sd.""")
tf.app.flags.DEFINE_integer('batch_size', 10,"""Number of images to process in a batch.""")

def computeNormalizationParameters():

  all_train_visible_idx = [x for x in xrange(0,FLAGS.total_visible_training_images)]
  random.shuffle(all_train_visible_idx)
  stats_sample_indexes = all_train_visible_idx[0:FLAGS.stats_sample_size]

  anno_file_batch_rows = []
  anno_file = open('cdhd_anno_training_data.txt')
  anno_file_lines = anno_file.readlines()

  for x in stats_sample_indexes:
    anno_file_batch_rows.append(anno_file_lines[x])  
  
  # mean_pixel = np.zeros(3);

  mean_pixel = np.zeros(3);
  mean_pixel[0] = 118.1365
  mean_pixel[1] = 114.5391
  mean_pixel[2] = 111.4741

  # mean_pixel_sq = np.zeros(3);
  
  mean_pixel_sq = np.zeros(3);
  mean_pixel_sq[0] = 19350.7498
  mean_pixel_sq[1] = 18537.0203
  mean_pixel_sq[2] = 18291.5741  

  pixel_covar = np.zeros((3, 3));

  '''
    TODO: It may be possible to use numpy apis to calculate mean and std dev directly
  '''

  '''
  num_pixel = 0
  for image_idx in xrange(FLAGS.stats_sample_size):
    image_filename = anno_file_batch_rows[image_idx].split('|')[2]
    image = Image.open(FLAGS.data_dir + image_filename)

    try:
      im = np.array(image.getdata()).reshape(image.size[0], image.size[1], 3)
    except ValueError:
      im = np.dstack([im]*3)

    #scale
    scale = round(FLAGS.max_im_side/float(np.amax(im.shape)),4)

    #reshape
    im = im.reshape(im.shape[0] * im.shape[1],im.shape[2])
    npix = im.shape[0]

    mean_pixel = mean_pixel * (float(num_pixel)/float(num_pixel + npix)) \
                   + np.sum(im, axis=0)/float(num_pixel + npix)
    mean_pixel_sq = mean_pixel_sq * (float(num_pixel) / float(num_pixel + npix)) \
                   + np.sum(im ** 2, axis=0)/float(num_pixel + npix)

    pixel_covar = pixel_covar * (float(num_pixel)/float(num_pixel + npix)) \
                   + (np.transpose(im).dot(im))/float(num_pixel + npix)

    num_pixel = num_pixel + npix;
  '''

  mean_pixel_113 = np.zeros((1,1,3))
  mean_pixel_113[0][0][0] = mean_pixel[0]
  mean_pixel_113[0][0][1] = mean_pixel[1]
  mean_pixel_113[0][0][2] = mean_pixel[2]

  mean_pixel_sq_113 = np.zeros((1,1,3))
  mean_pixel_sq_113[0][0][0] = mean_pixel_sq[0]
  mean_pixel_sq_113[0][0][1] = mean_pixel_sq[1]
  mean_pixel_sq_113[0][0][2] = mean_pixel_sq[2]

  # std_pixel = np.sqrt(mean_pixel_sq - (mean_pixel ** 2))
  # stats_dict = {'mean_pixel': mean_pixel, 'std_pixel': std_pixel, 'pixel_covar': pixel_covar}

  std_pixel = np.sqrt(mean_pixel_sq_113 - (mean_pixel_113 ** 2))
  stats_dict = {'mean_pixel': mean_pixel_113, 'std_pixel': std_pixel, 'pixel_covar': pixel_covar}
  

  #store values so that there's no need to compute next time

  return stats_dict

def getImageMetaRecords():
  all_train_visible_idx = [x for x in xrange(0,FLAGS.total_visible_training_images)]
  random.shuffle(all_train_visible_idx)
  # batch_indexes = all_train_visible_idx[0:batch_size]

  anno_file_batch_rows = []
  anno_file = open('cdhd_anno_training_data.txt')
  anno_file_lines = anno_file.readlines()

  for x in all_train_visible_idx:
    anno_file_batch_rows.append(anno_file_lines[x])

  return anno_file_batch_rows

def train(stats_dict):
  """
  tf.Graph().as_default():
  This method should be used if you want to create multiple graphs in the same process. 
  For convenience, a global default graph is provided, and all ops will be added to this graph 
  if you do not create a new graph explicitly. Use this method with the with keyword to specify 
  that ops created within the scope of a block should be added to this graph.
  """

  with tf.Graph().as_default():
    global_step = tf.Variable(0, trainable=False)
    images = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, None, None, 3])
    out_locs = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    org_gt_coords = tf.placeholder(dtype=tf.float32, shape=[FLAGS.batch_size, 2])    

    # Get images and labels for CDHD.
    #images, meta = cdhd.distorted_inputs(stats_dict) # move this to the for loop

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cdhd.buildModelAndTrain(images,out_locs,org_gt_coords)

    #import pdb
    #pdb.set_trace()

    '''
  # Calculate loss. 
    loss = cdhd.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cdhd.train(loss, global_step)      

    #example input: 163.3941|99.1765|car_ims/000002.jpg|219,460,3|1|0|1|1|48,24,441,202
    
    #global all_training_data
    CDHDGlobals.all_training_data = []
    '''

    # with tf.Session() as sess:
    #   sess.run(logits) #feed_dict, key = node, value = numpy array
    #   print('end')

    init = tf.global_variables_initializer()
    sess = tf.Session()
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    sess.run(init)

    # Following two lines are for debugging
    # Use <code> python cdhd_train.py --debug/ <code> command to debug
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    for epoch in xrange(FLAGS.max_steps):
    # for epoch in xrange(200):
      start_time = time.time()
      anno_file_batch_rows = getImageMetaRecords() 
      print('epoch: ', epoch)

      for batch in xrange(len(anno_file_batch_rows)/FLAGS.batch_size):
        # distorted_images, meta = cdhd_input.distorted_inputs(stats_dict, FLAGS.batch_size)
        distorted_images, meta = cdhd_input.distorted_inputs(stats_dict, FLAGS.batch_size, \
                anno_file_batch_rows[batch * FLAGS.batch_size : (batch * FLAGS.batch_size) + FLAGS.batch_size])

        # print('images shape: ',distorted_images.shape)

        loss = sess.run(logits, {images: distorted_images, out_locs: meta['out_locs'], \
                org_gt_coords: meta['org_gt_coords']})

        # print('loss shape: ', loss.shape)

        print(batch, np.sum(loss, axis=0)[0,0,0])

      duration = time.time() - start_time

      # print('duration: ', duration)
    
    writer.close()

def main(argv=None):  # pylint: disable=unused-argument
  stats_dict = computeNormalizationParameters()
  train(stats_dict)

if __name__ == '__main__':
  tf.app.run()    