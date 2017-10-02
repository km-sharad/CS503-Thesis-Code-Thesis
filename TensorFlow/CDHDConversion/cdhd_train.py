import cdhd
import random
import tensorflow as tf
from PIL import Image
from math import sqrt
import numpy as np
import time

FLAGS = tf.app.flags.FLAGS

#server
#tf.app.flags.DEFINE_string('data_dir', '/home/sharad/CS503-Thesis/car_dataset/', """Path to the CDHD data directory.""")

#local
tf.app.flags.DEFINE_string('data_dir', '../../../../car_dataset/', """Path to the CDHD data directory.""")
tf.app.flags.DEFINE_integer('total_visible_training_images', 1920,
                              """Number of training images where car door handle is visible.""")
tf.app.flags.DEFINE_integer('max_steps', 26000,"""Number of batches to run.""")
tf.app.flags.DEFINE_integer('stats_sample_size', 200,"""Number of images to calculate mean and sd.""")

def computeNormalizationParameters():

  all_train_visible_idx = [x for x in xrange(0,FLAGS.total_visible_training_images)]
  random.shuffle(all_train_visible_idx)
  stats_sample_indexes = all_train_visible_idx[0:FLAGS.stats_sample_size]

  anno_file_batch_rows = []
  anno_file = open('cdhd_anno_training_data.txt')
  anno_file_lines = anno_file.readlines()

  for x in stats_sample_indexes:
    anno_file_batch_rows.append(anno_file_lines[x])  
  
  #mean_pixel = np.zeros(3);

  mean_pixel = np.zeros(3);
  mean_pixel[0] = 118.1365
  mean_pixel[1] = 114.5391
  mean_pixel[2] = 111.4741

  #mean_pixel_sq = np.zeros(3);
  
  mean_pixel_sq = np.zeros(3);
  mean_pixel_sq[0] = 19350.7498
  mean_pixel_sq[1] = 18537.0203
  mean_pixel_sq[2] = 18291.5741  

  pixel_covar = np.zeros((3, 3));

  '''
    TODO: It may be possible to use numpy apis to calculater mean and std dev directly
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

  std_pixel = np.sqrt(mean_pixel_sq - (mean_pixel ** 2))
  
  stats_dict = {'mean_pixel': mean_pixel_113, 'std_pixel': std_pixel, 'pixel_covar': pixel_covar}

  #store values so that there's no need to compute next time

  return stats_dict

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

    # Get images and labels for CDHD.
    #images, meta = cdhd.distorted_inputs(stats_dict) # move this to the for loop

    # Build a Graph that computes the logits predictions from the
    # inference model.
    #logits = cdhd.inference(images, meta) #should be build first outside the for loop. for loop samples images ans passes tothe graph
    logits = cdhd.inference()
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

    with tf.Session() as sess:
      sess.run(logits) #feed_dict, key = node, value = numpy array
      print('end')

    '''
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()
      _, loss_value = sess.run([train_op, loss])
      duration = time.time() - start_time
    '''

def main(argv=None):  # pylint: disable=unused-argument
  stats_dict = computeNormalizationParameters()
  train(stats_dict)

if __name__ == '__main__':
  tf.app.run()    