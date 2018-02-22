import dwi
import random
import tensorflow as tf
from PIL import Image
from math import sqrt
import numpy as np
import time
import dwi_input
from tensorflow.python import debug as tf_debug
import sys
from scipy.misc import imresize

data_dir = 'dog_walking_dataset/'
total_visible_training_images = 250    # Number of training images 
total_visible_test_images = 70         # Number of validation images 
stats_sample_size = 100                # Number of images to calculate mean and sd
batch_size = 10                        # Number of images to process in a batch
max_im_side = 500

def computeNormalizationParameters():
  all_train_visible_idx = [x for x in xrange(0,total_visible_training_images)]
  random.shuffle(all_train_visible_idx)
  stats_sample_indexes = all_train_visible_idx[0:stats_sample_size]

  anno_file_batch_rows = []
  anno_file = open('dwi_anno_training_data.txt')
  anno_file_lines = anno_file.readlines()

  for x in stats_sample_indexes:
    anno_file_batch_rows.append(anno_file_lines[x])
 
  mean_pixel = np.zeros(3);
  mean_pixel_sq = np.zeros(3);
  pixel_covar = np.zeros((3, 3));

  num_pixel = 0
  for image_idx in xrange(stats_sample_size):
    image_filename = anno_file_batch_rows[image_idx].split('|')[4]
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

  return stats_dict

def getTestImageMetaRecords():
  all_test_visible_idx = [x for x in xrange(0, total_visible_test_images)]
  random.shuffle(all_test_visible_idx)

  test_anno_file_batch_rows = []
  test_anno_file = open('dwi_anno_testing_data.txt')
  test_anno_file_lines = test_anno_file.readlines()  

  for x in all_test_visible_idx:
    test_anno_file_batch_rows.append(test_anno_file_lines[x])  

  return test_anno_file_batch_rows  

def calculateIOU(pred_0, pred_1, original_0, original_1, bbox):
  pred_0 = pred_0.reshape(batch_size,2)
  #following is necessary since NHWC format of tf has height followed by width
  #it's therefore necessary to flip it since in annotation, the first coordinates is x followed by y
  #Also, the first coordinate in the bbox is x-coordinate of the top left corner of bounding box 
  #and the second is y-coordinate
  #For more info: 'Data formats' section in https://www.tensorflow.org/performance/performance_guide#use_nchw_imag
  pred_0 = np.flip(pred_0,1)      

  pred_1 = pred_1.reshape(batch_size,2)
  pred_1 = np.flip(pred_1, 1)

  #x0, y0 = top left coordinates, x1, y1 = bottom right coordinate
  pred_coords = {}
  actual_coords = {}

  iou_threshold = 0.5
  cases_above_iou_threshold = 0

  # print('bbox: ', bbox)

  for i in xrange(pred_0.shape[0]): 
    pred_0_x = int((pred_0[i])[0])
    pred_0_y = int((pred_0[i])[1])
    pred_1_x = int((pred_1[i])[0])
    pred_1_y = int((pred_1[i])[1])    

    if(pred_0_x < pred_1_x):
      pred_coords['x0'] = pred_0_x
      pred_coords['x1'] = pred_1_x
      if(pred_0_y < pred_1_y):
        pred_coords['y0'] = pred_0_y
        pred_coords['y1'] = pred_1_y
      elif(pred_0_y > pred_1_y):
        pred_coords['y0'] = pred_1_y
        pred_coords['y1'] = pred_0_y
      else:
        # pred_0_y is same as pred_1_y, intersection area cannot be calculated
        continue
    elif(pred_0_x > pred_1_x):
      pred_coords['x0'] = pred_1_x
      pred_coords['x1'] = pred_0_x
      if(pred_0_y < pred_1_y):
        pred_coords['y0'] = pred_0_y
        pred_coords['y1'] = pred_1_y      
      elif(pred_0_y > pred_1_y):
        pred_coords['y0'] = pred_1_y
        pred_coords['y1'] = pred_0_y
      else:
        # pred_0_y is same as pred_1_y, intersection area cannot be calculated
        continue        
    else:
      # pred_0_x is same as pred_1_x, intersection area cannot be calculated
      print('pred_0_x: ', pred_0_x, pred_1_x)
      continue

    assert pred_coords['x0'] < pred_coords['x1']
    assert pred_coords['y0'] < pred_coords['y1']

    actual_coords['x0'] = (bbox[i])[0]
    actual_coords['y0'] = (bbox[i])[1]
    actual_coords['x1'] = (bbox[i])[2]
    actual_coords['y1'] = (bbox[i])[3]

    assert actual_coords['x0'] < actual_coords['x1']
    assert actual_coords['y0'] < actual_coords['y1']    

    # determine the coordinates of the intersection rectangle
    x_left = max(pred_coords['x0'], actual_coords['x0'])
    y_top = max(pred_coords['y0'], actual_coords['y0'])
    x_right = min(pred_coords['x1'], actual_coords['x1'])
    y_bottom = min(pred_coords['y1'], actual_coords['y1']) 

    if x_right < x_left or y_bottom < y_top:
      intersection_area = 0
    else:           
      intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both predicted and actual bounding boxes
    pred_area = (pred_coords['x1'] - pred_coords['x0']) * (pred_coords['y1'] - pred_coords['y0'])
    actual_area = (actual_coords['x1'] - actual_coords['x0']) * (actual_coords['y1'] - actual_coords['y0'])    

    iou = intersection_area/float(pred_area + actual_area - intersection_area)
    out_f_iou = open('out_test_iou.txt', 'a+')
    out_f_iou.write('iou: ' + str(iou) + '\n')
    out_f_iou.close()                
    assert (0.0 <= iou <= 1.0)

    if(iou >= iou_threshold):
      cases_above_iou_threshold = cases_above_iou_threshold + 1

  batch_accuracy = cases_above_iou_threshold/float(pred_0.shape[0])
  return batch_accuracy

global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
images = tf.placeholder(dtype=tf.float32, shape=[batch_size, None, None, 3])
out_locs = tf.placeholder(dtype=tf.float32, shape=[None, 2])
org_gt_coords_ll0 = tf.placeholder(dtype=tf.float32, shape=[batch_size, 2])   
org_gt_coords_ll1 = tf.placeholder(dtype=tf.float32, shape=[batch_size, 2])   

stats_dict = computeNormalizationParameters() 

res_aux = dwi.inference(images, out_locs, org_gt_coords_ll0, org_gt_coords_ll1)

ret_dict = dwi.test(res_aux, global_step)

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=20)

with tf.Session() as sess:
  writer = tf.summary.FileWriter('./graphs', sess.graph)
  sess.run(init)

  # Restore variables from disk.
  saver.restore(sess, "./ckpt/model2.ckpt")
  print("Model restored.")

  anno_file_batch_rows = getTestImageMetaRecords()

  # for batch in xrange(5): 
  for batch in xrange(len(anno_file_batch_rows)/batch_size):
    test_images, test_meta = dwi_input.distorted_inputs(stats_dict, batch_size, \
            anno_file_batch_rows[batch * batch_size : (batch * batch_size) + batch_size])      

    test_dict = sess.run(ret_dict, feed_dict=
                      {images: test_images, 
                      out_locs: test_meta['out_locs'],
                      org_gt_coords_ll0: test_meta['org_gt_coords_ll0'],
                      org_gt_coords_ll1: test_meta['org_gt_coords_ll1']})

    batch_accuracy = calculateIOU(test_dict['pred_coord_0'], 
                  test_dict['pred_coord_1'],
                  test_meta['org_gt_coords_ll0'],
                  test_meta['org_gt_coords_ll1'],
                  test_meta['bbox'])

    out_f = open('out_test_file.txt', 'a+')
    out_f.write(str(batch) + ' ' + str(batch_accuracy))
    out_f.close()        
    print('batch: ', batch, batch_accuracy)

