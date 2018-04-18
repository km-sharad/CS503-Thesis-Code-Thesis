import cdhd
import random
import tensorflow as tf
from PIL import Image
from math import sqrt
import numpy as np
import time
import cdhd_input
from tensorflow.python import debug as tf_debug
import sys
from scipy.misc import imresize
import vgg16

data_dir = '../../../../../../CS503-Thesis/car_dataset/'
#data_dir = '../../../../car_dataset/'
total_visible_training_images = 1920    # Number of training images where car door handle is visible
total_visible_test_images = 1200 # Number of validation images where car door handle is visible
stats_sample_size = 200                 # Number of images to calculate mean and sd
batch_size = 10                         # Number of images to process in a batch
max_im_side = 500

def computeNormalizationParameters():

  all_train_visible_idx = [x for x in xrange(0,total_visible_training_images)]
  random.shuffle(all_train_visible_idx)
  stats_sample_indexes = all_train_visible_idx[0:stats_sample_size]

  anno_file_batch_rows = []
  anno_file = open('cdhd_anno_training_data.txt')
  anno_file_lines = anno_file.readlines()

  for x in stats_sample_indexes:
    anno_file_batch_rows.append(anno_file_lines[x])
 
  mean_pixel = np.zeros(3);
  mean_pixel_sq = np.zeros(3);
  pixel_covar = np.zeros((3, 3));

  num_pixel = 0
  for image_idx in xrange(stats_sample_size):
    image_filename = anno_file_batch_rows[image_idx].split('|')[2]
    image = Image.open(data_dir + image_filename)

    im = np.array(image, dtype=np.uint64)
    if(len(im.shape) == 2):
      #monochrome image, add the third channel
      im = np.stack((image,)*3)
      print('monochrome image for mean calc: ', image_filename)      

    #scale
    scale = round(max_im_side/float(np.amax(im.shape)),4)
    im = imresize(im, scale, interp='bilinear')
    im = im.astype(np.uint64)

    #reshape
    im = im.reshape(im.shape[0] * im.shape[1],im.shape[2])
    npix = im.shape[0]

    mean_pixel = mean_pixel * (float(num_pixel)/(float(num_pixel + npix))) \
                    + np.sum(im, axis=0)/((float(num_pixel + npix)))
    mean_pixel_sq = mean_pixel_sq * (float(num_pixel)/(float(num_pixel + npix))) \
                    + np.sum(np.square(im), axis=0)/(float(num_pixel + npix))

    num_pixel = num_pixel + npix;

  epsilon = 0.001;
  std_pixel = np.sqrt(mean_pixel_sq - np.square(mean_pixel)) + epsilon
  stats_dict = {'mean_pixel': mean_pixel, 'std_pixel': std_pixel}

  #store values so that there's no need to compute next time    
  

  return stats_dict

def getTestImageMetaRecords():
  all_test_visible_idx = [x for x in xrange(0, total_visible_test_images)]
  random.shuffle(all_test_visible_idx)

  test_anno_file_batch_rows = []
  test_anno_file = open('cdhd_anno_testing_data.txt')
  test_anno_file_lines = test_anno_file.readlines()  

  for x in all_test_visible_idx:
    test_anno_file_batch_rows.append(test_anno_file_lines[x])  

  return test_anno_file_batch_rows  

def calculteNormalizedTestDistance(pred, original, bbox_heights):
  total_normalized_distance = 0
  for i in xrange(original.shape[0]): 
    # print(pred[i][0][0][0],pred[i][0][1][0],original[i][0], original[i][1], bbox_heights[i])    
    mse = sqrt(pow((pred[i][0][0][0] - original[i][0]), 2) + pow((pred[i][0][1][0] - original[i][1]), 2))
    normalized_dist = mse/float(bbox_heights[i])    
    # print(normalized_dist)
    total_normalized_distance = total_normalized_distance + normalized_dist

  # return average normalized distance for the batch of 10 images
  return total_normalized_distance/float(original.shape[0])

global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
images = tf.placeholder(dtype=tf.float32, shape=[batch_size, 224, 224, 3])
out_locs = tf.placeholder(dtype=tf.float32, shape=[None, 2])
org_gt_coords = tf.placeholder(dtype=tf.float32, shape=[batch_size, 2])  

stats_dict = computeNormalizationParameters() 

vgg = vgg16.Vgg16()
vgg.build(images)

# res_aux = cdhd.inference(images,out_locs,org_gt_coords)
res_aux = cdhd.doForwardPass(vgg.pool4, out_locs, org_gt_coords)

val_dict = cdhd.test(res_aux, global_step)

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=1)

with tf.Session() as sess:
  writer = tf.summary.FileWriter('./graphs', sess.graph)
  sess.run(init)

  # Restore variables from disk.
  saver.restore(sess, "./ckpt/model27.ckpt")
  print("Model restored.")

  # Following two lines are for debugging
  # Use <code> python cdhd_train.py --debug </code> command to debug
  # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
  # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

  # all_train_visible_idx = [x for x in xrange(0,total_visible_training_images)]
  # random.shuffle(all_train_visible_idx)  

  for norm_dist in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]:
    print('norm_dist: ', norm_dist)
    total_batch = 0
    batch_le_norm_dist = 0

    anno_file_batch_rows = getTestImageMetaRecords()

    for batch in xrange(len(anno_file_batch_rows)/batch_size):
      test_images, test_meta = cdhd_input.distorted_inputs(stats_dict, batch_size, \
              anno_file_batch_rows[batch * batch_size : (batch * batch_size) + batch_size])      

      test_dict = sess.run(val_dict, feed_dict =
                                    {images: test_images, 
                                    out_locs: test_meta['out_locs'],
                                    org_gt_coords: test_meta['org_gt_coords']})  
      avg_normalized_dist = calculteNormalizedTestDistance(test_dict['pred_coord'], 
                                            test_meta['org_gt_coords'],
                                            test_meta['bbox_heights'])

      total_batch = total_batch + 1
      if(round(avg_normalized_dist, 2) <= norm_dist):
        batch_le_norm_dist = batch_le_norm_dist + 1

    print(norm_dist, float(batch_le_norm_dist)/float(total_batch))
    out_f = open('out_test_file.txt', 'a+')
    out_f.write(str(norm_dist) + ' ' + str(float(batch_le_norm_dist)/float(total_batch)) + '\n')
    out_f.close()    
    
  writer.close() 
