import dwi
import dwi_input
import random
import tensorflow as tf
from PIL import Image
from math import sqrt
import numpy as np
import time
from scipy.misc import imresize
from tensorflow.python import debug as tf_debug

data_dir = 'dog_walking_dataset/'
total_visible_training_images = 250     # Number of training images 
total_visible_validation_images = 60    # Number of validation images 
max_steps = 5000                        # Number of batches to run
stats_sample_size = 100                 # Number of images to calculate mean and sd
batch_size = 10                         # Number of images to process in a batch
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

def getValidationImageMetaRecords():
  all_val_visible_idx = [x for x in xrange(0, total_visible_validation_images)]
  random.shuffle(all_val_visible_idx)

  val_anno_file_batch_rows = []
  val_anno_file = open('dwi_anno_val_data.txt')
  val_anno_file_lines = val_anno_file.readlines()  

  for x in all_val_visible_idx:
    val_anno_file_batch_rows.append(val_anno_file_lines[x])  

  return val_anno_file_batch_rows[0:batch_size]    

def getImageMetaRecords():
  all_train_visible_idx = [x for x in xrange(0,total_visible_training_images)]
  random.shuffle(all_train_visible_idx)
  # batch_indexes = all_train_visible_idx[0:batch_size]

  anno_file_batch_rows = []
  anno_file = open('dwi_anno_training_data.txt')
  anno_file_lines = anno_file.readlines()

  for x in all_train_visible_idx:
    anno_file_batch_rows.append(anno_file_lines[x])

  return anno_file_batch_rows

def calculateIOU(pred_0, pred_1, original_0, original_1, bbox):
  print('original_0: ', original_0)
  print('original_1: ', original_1)
  pred_0 = pred_0.reshape(batch_size,2)
  print('pred_0: ', pred_0)
  #following is necessary since NHWC format of tf has height followed by width
  #it's therefore necessary to flip it since in annotation, the first coordinates is x followed by y
  #Also, the first coordinate in the bbox is x-coordinate of the top left corner of bounding box 
  #and the second is y-coordinate
  #For more info: 'Data formats' section in https://www.tensorflow.org/performance/performance_guide#use_nchw_imag
  pred_0 = np.flip(pred_0,1)      

  pred_1 = pred_1.reshape(batch_size,2)
  print('pred_1: ', pred_1)
  pred_1 = np.flip(pred_1, 1)

  #x0, y0 = top left coordinates, x1, y1 = bottom right coordinate
  pred_coords = {}
  actual_coords = {}

  iou_threshold = 0.5
  cases_above_iou_threshold = 0

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

    # print('pred: ', pred_coords['x0'], pred_coords['y0'], pred_coords['x1'], pred_coords['y1'])
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
    out_f_iou = open('out_val_iou.txt', 'a+')
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

ret_dict = dwi.train(res_aux, global_step)

val_dict = dwi.test(res_aux, global_step)

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=20)

with tf.Session() as sess:
  writer = tf.summary.FileWriter('./graphs', sess.graph)
  sess.run(init)

  # Restore variables from disk.
  # saver.restore(sess, "./ckpt/model0.ckpt")
  # print("Model restored.")

  # Following two lines are for debugging
  # Use <code> python cdhd_train.py --debug </code> command to debug
  # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
  # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

  for epoch in xrange(max_steps):
    start_time = time.time()
    anno_file_batch_rows = getImageMetaRecords() 
    print('epoch: ', epoch)

    for batch in xrange(len(anno_file_batch_rows)/batch_size):
      distorted_images, meta = dwi_input.distorted_inputs(stats_dict, batch_size, \
              anno_file_batch_rows[batch * batch_size : (batch * batch_size) + batch_size])

      out_dict = sess.run(ret_dict, feed_dict=
                            {images: distorted_images, 
                            out_locs: meta['out_locs'],
                            org_gt_coords_ll0: meta['org_gt_coords_ll0'],
                            org_gt_coords_ll1: meta['org_gt_coords_ll1']})

      # print('global_step: %s' % tf.train.global_step(sess, global_step))
      # print('poc_shape: ', out_dict['poc_shape'])

      out_f = open('out_file.txt', 'a+')
      out_f.write(str(epoch) + ' ' + str(batch) + ' ' + str(out_dict['loss']) + '\n')
      out_f.close()

    # Save the variables to disk.
    ckpt_file = './ckpt/model' + str(epoch) + '.ckpt'
    save_path = saver.save(sess, ckpt_file)
    print("Model saved in file: %s" % save_path)

    out_f_epoch = open('out_epoch.txt', 'a+')
    out_f_epoch.write(str(epoch) + ' ' + str(out_dict['loss']) + '\n')
    out_f_epoch.close()        

    # Validation step
    if((epoch % 2) == 0):
      validation_images, validation_meta = dwi_input.distorted_inputs(stats_dict, batch_size, getValidationImageMetaRecords())

      validation_dict = sess.run(val_dict, feed_dict=
                        {images: validation_images, 
                        out_locs: validation_meta['out_locs'],
                        org_gt_coords_ll0: validation_meta['org_gt_coords_ll0'],
                        org_gt_coords_ll1: validation_meta['org_gt_coords_ll1']})      

      batch_accuracy = calculateIOU(validation_dict['pred_coord_0'], 
                    validation_dict['pred_coord_1'],
                    validation_meta['org_gt_coords_ll0'],
                    validation_meta['org_gt_coords_ll1'],
                    validation_meta['bbox'])

      out_f_validation = open('validation_epoch.txt', 'a+')
      out_f_validation.write(str(epoch) + ' ' + str(batch_accuracy) + '\n')
      out_f_validation.close()             

  writer.close() 
