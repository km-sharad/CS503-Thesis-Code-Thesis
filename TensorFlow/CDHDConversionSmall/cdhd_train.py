import cdhd
import random
import tensorflow as tf
from PIL import Image
from math import sqrt
import numpy as np
import time
import cdhd_input
from tensorflow.python import debug as tf_debug

total_visible_training_images = 1920    # Number of training images where car door handle is visible
max_steps = 26000                       # Number of batches to run
stats_sample_size = 200                 # Number of images to calculate mean and sd
batch_size = 10                         # Number of images to process in a batch

def computeNormalizationParameters():

  all_train_visible_idx = [x for x in xrange(0,total_visible_training_images)]
  random.shuffle(all_train_visible_idx)
  stats_sample_indexes = all_train_visible_idx[0:stats_sample_size]

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
  for image_idx in xrange(stats_sample_size):
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
  all_train_visible_idx = [x for x in xrange(0,total_visible_training_images)]
  random.shuffle(all_train_visible_idx)
  # batch_indexes = all_train_visible_idx[0:batch_size]

  anno_file_batch_rows = []
  anno_file = open('cdhd_anno_training_data.txt')
  anno_file_lines = anno_file.readlines()

  for x in all_train_visible_idx:
    anno_file_batch_rows.append(anno_file_lines[x])

  return anno_file_batch_rows

global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
images = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, None, 3])
out_locs = tf.placeholder(dtype=tf.float32, shape=[None, 2])
org_gt_coords = tf.placeholder(dtype=tf.float32, shape=[batch_size, 2])   

stats_dict = computeNormalizationParameters() 

# logits = cdhd.buildModelAndTrain(images,out_locs,org_gt_coords, global_step)

res_aux = cdhd.inference(images,out_locs,org_gt_coords)

ret_dict = cdhd.train(res_aux, global_step)

# with tf.Graph().as_default():
init = tf.global_variables_initializer()
with tf.Session() as sess:
  writer = tf.summary.FileWriter('./graphs', sess.graph)
  sess.run(init)

  # Following two lines are for debugging
  # Use <code> python cdhd_train.py --debug </code> command to debug
  # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
  # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

  for epoch in xrange(max_steps):
    start_time = time.time()
    anno_file_batch_rows = getImageMetaRecords() 
    print('epoch: ', epoch)

    for batch in xrange(len(anno_file_batch_rows)/batch_size):
      distorted_images, meta = cdhd_input.distorted_inputs(stats_dict, batch_size, \
              anno_file_batch_rows[batch * batch_size : (batch * batch_size) + batch_size])

      out_dict = sess.run(ret_dict, feed_dict=
                            {images: distorted_images, 
                            out_locs: meta['out_locs'],
                            org_gt_coords: meta['org_gt_coords']})

      # print('global_step: %s' % tf.train.global_step(sess, global_step))

      # print('loss shape: ', out_dict['loss'].shape)
      print(batch, np.sum(out_dict['loss'], axis=0)[0,0,0])
      
      # for idx1 in xrange(len(out_dict['grad_var'])):
      #   print('grad_var shape idx1: ', out_dict['grad_var'][idx1][0].shape, out_dict['grad_var'][idx1][1].shape)

      # for idx_a in xrange(len(out_dict['grad_var'])):
      #   assert not np.isnan(out_dict['grad_var'][idx_a][0].all()), '*** NaN gradient'
      #   assert not np.isinf(out_dict['grad_var'][idx_a][0].all()), '*** INF gradient'       

      # for idx in xrange(len(out_dict['grad_var'])):
      #   if idx < 3:
      #     print('gradient val var: ', out_dict['grad_var'][idx][0][2][3][87][24])
      #     print('variable val var: ', out_dict['grad_var'][idx][1][2][3][87][24]) 
      #     print('weights_col2-before: ', out_dict['weights_col2_before'][idx][2][3][87][24])           
      #     print('weights_col2-after: ', out_dict['weights_col2_after'][idx][2][3][87][24])           
      #   else:
      #     print('gradient val bias: ', out_dict['grad_var'][idx][0][21])
      #     print('variable val bias: ', out_dict['grad_var'][idx][1][21])                                

    duration = time.time() - start_time

  writer.close() 
